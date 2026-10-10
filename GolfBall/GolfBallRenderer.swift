//
//  GolfBallRenderer.swift
//
//  Host side for GolfBall.metal:
//    1. Create the MTLDevice, command queue and compute pipeline state.
//    2. Allocate unified-memory (.storageModeShared) MTLBuffers for the
//       uniforms and for the output image.
//    3. Dispatch the kernel with a SIMD-aligned 2D threadgroup layout.
//    4. Read the pixels back on the CPU (no blit, no GPU-side copy).
//
//  Zero-copy readback: the output texture is a *view* onto a shared
//  MTLBuffer (`MTLBuffer.makeTexture`). On Apple silicon the CPU and GPU
//  share that memory, so once the command buffer completes,
//  `outputBuffer.contents()` already holds the rendered pixels.
//
//  The scene is static (Sunday at the Masters, player at address), so it is
//  rendered once at high quality rather than every display frame. The image
//  is rendered in horizontal bands, one command buffer each, so no single GPU
//  submission runs long enough to trip the macOS GPU watchdog.
//

import Metal
import simd
import CoreGraphics
import SwiftUI

// MARK: - Uniforms (must match `GolfSceneUniforms` in GolfBall.metal) -----------

struct GolfSceneUniforms {
    /// Camera position in cm; w = lens aperture radius in cm (0 = pinhole).
    /// Target line is +X; the right-handed player stands on the -Z side.
    var cameraPosition = AddressView.broadcast.cameraPosition
    /// Look-at point in cm; w = focal length (larger = narrower field of view).
    var cameraTarget   = AddressView.broadcast.cameraTarget
    /// Distance from the camera to the plane in perfect focus, in cm.
    var focusDistance: Float = AddressView.broadcast.focusDistance
    var samplesPerAxis: UInt32 = 4      // 16 samples: anti-aliasing + depth of field
    var dimpleCount:    UInt32 = 392
    var debugTiles:     UInt32 = 0      // 1 = tint threadgroup tiles
    var padding = SIMD4<UInt32>(repeating: 0)
}

/// Camera presets.
enum AddressView: String, CaseIterable, Identifiable {
    case broadcast = "Address"        // face-on TV angle: player, club and ball in focus
    case clubface  = "Clubface"       // low, from the target side: lofted face behind the ball
    case back      = "T100 Back"      // from behind the toe: cavity badge and markings

    var id: Self { self }

    var cameraPosition: SIMD4<Float> {
        switch self {
        case .broadcast: SIMD4(-60.0, 98.0, 1100.0, 5.0)
        case .clubface:  SIMD4(20.0, 8.0, 21.0, 0.25)
        case .back:      SIMD4(-21.0, 9.5, 13.0, 0.15)
        }
    }

    var cameraTarget: SIMD4<Float> {
        switch self {
        case .broadcast: SIMD4(-30.0, 86.0, -90.0, 9.0)
        case .clubface:  SIMD4(-1.2, 3.5, -0.5, 2.0)
        case .back:      SIMD4(-2.6, 2.3, 0.4, 2.4)
        }
    }

    var focusDistance: Float {
        let eye = SIMD3(cameraPosition.x, cameraPosition.y, cameraPosition.z)
        let focus: SIMD3<Float> = switch self {
        case .broadcast: SIMD3(0.0, 70.0, -55.0)     // hands and lead thigh
        case .clubface:  SIMD3(0.0, 3.5, 0.0)        // ball
        case .back:      SIMD3(-2.6, 1.6, 0.4)       // back of the head
        }
        let target = SIMD3(cameraTarget.x, cameraTarget.y, cameraTarget.z)
        return simd_dot(focus - eye, simd_normalize(target - eye))
    }
}

/// Carries a value the compiler can't prove Sendable across a completion handler.
private struct UncheckedSendable<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

enum GolfBallRendererError: Error {
    case noMetalDevice
    case noCommandQueue
    case functionNotFound(String)
    case bufferAllocationFailed
    case textureCreationFailed
    case commandBufferFailed(Error?)
}

// MARK: - Renderer --------------------------------------------------------------

/// Renders the scene into CPU-readable unified memory.
///
/// Renders must not overlap (await each one), because the output buffer is reused.
final class GolfBallRenderer: @unchecked Sendable {

    let device: MTLDevice
    let width: Int
    let height: Int
    let bytesPerRow: Int

    var uniforms = GolfSceneUniforms()

    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let uniformBuffer: MTLBuffer     // shared: CPU writes, GPU reads
    private let tileOriginBuffer: MTLBuffer  // shared: one band origin per 256-byte slot
    private let outputBuffer: MTLBuffer      // shared: GPU writes, CPU reads
    private let outputTexture: MTLTexture    // linear texture view onto outputBuffer

    /// Threads per threadgroup: one SIMD-group wide, as tall as the pipeline allows.
    let threadgroupSize: MTLSize
    /// Rows per band (a multiple of the threadgroup height).
    let bandHeight: Int
    private var bandCount: Int { (height + bandHeight - 1) / bandHeight }

    private static let kernelName = "golfSceneKernel"
    private static let pixelFormat: MTLPixelFormat = .bgra8Unorm
    private static let bytesPerPixel = 4
    /// Constant-buffer offsets must be 256-byte aligned on some Macs; use that everywhere.
    private static let tileOriginStride = 256

    /// - Parameter library: pass a library explicitly (e.g. a .metallib in tests);
    ///   defaults to the app bundle's default library.
    init(width: Int, height: Int,
         device: MTLDevice? = MTLCreateSystemDefaultDevice(),
         library: MTLLibrary? = nil) throws {

        // 1. Device, queue, pipeline -------------------------------------------
        guard let device else { throw GolfBallRendererError.noMetalDevice }
        guard let queue = device.makeCommandQueue() else { throw GolfBallRendererError.noCommandQueue }

        let library = try library ?? device.makeDefaultLibrary(bundle: .main)
        guard let function = library.makeFunction(name: Self.kernelName) else {
            throw GolfBallRendererError.functionNotFound(Self.kernelName)
        }
        let pipeline = try device.makeComputePipelineState(function: function)

        // 2. Unified-memory buffers ---------------------------------------------
        // Linear textures need each row aligned to the device's requirement.
        let alignment   = device.minimumLinearTextureAlignment(for: Self.pixelFormat)
        let bytesPerRow = (width * Self.bytesPerPixel + alignment - 1) / alignment * alignment

        guard
            let uniformBuffer = device.makeBuffer(length: MemoryLayout<GolfSceneUniforms>.stride,
                                                  options: .storageModeShared),
            let outputBuffer  = device.makeBuffer(length: bytesPerRow * height,
                                                  options: .storageModeShared)
        else { throw GolfBallRendererError.bufferAllocationFailed }
        uniformBuffer.label = "GolfScene Uniforms"
        outputBuffer.label  = "GolfScene Output"

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: Self.pixelFormat,
                                                                  width: width, height: height,
                                                                  mipmapped: false)
        descriptor.storageMode = .shared
        descriptor.usage = [.shaderWrite, .shaderRead]
        guard let outputTexture = outputBuffer.makeTexture(descriptor: descriptor, offset: 0,
                                                           bytesPerRow: bytesPerRow)
        else { throw GolfBallRendererError.textureCreationFailed }

        // 3. Threadgroup layout -------------------------------------------------
        // Width = one SIMD-group (32 on Apple GPUs); height = as many rows as the
        // pipeline allows, which shrinks for register-heavy kernels like this ray
        // marcher. The pipeline reports the real limit, so nothing is hard-coded.
        let w = pipeline.threadExecutionWidth
        let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)

        // Bands of whole threadgroup rows, ~64 px tall, each with its own origin.
        let bandHeight = max(h, 64 / h * h)
        let bandCount  = (height + bandHeight - 1) / bandHeight
        guard let tileOriginBuffer = device.makeBuffer(length: bandCount * Self.tileOriginStride,
                                                       options: .storageModeShared)
        else { throw GolfBallRendererError.bufferAllocationFailed }
        tileOriginBuffer.label = "GolfScene Band Origins"
        for band in 0..<bandCount {
            tileOriginBuffer.contents()
                .advanced(by: band * Self.tileOriginStride)
                .storeBytes(of: SIMD2<UInt32>(0, UInt32(band * bandHeight)), as: SIMD2<UInt32>.self)
        }

        self.device = device
        self.width = width
        self.height = height
        self.bytesPerRow = bytesPerRow
        self.commandQueue = queue
        self.pipeline = pipeline
        self.uniformBuffer = uniformBuffer
        self.tileOriginBuffer = tileOriginBuffer
        self.outputBuffer = outputBuffer
        self.outputTexture = outputTexture
        self.threadgroupSize = MTLSize(width: w, height: h, depth: 1)
        self.bandHeight = bandHeight
    }

    /// Renders the scene and returns it as a CGImage (pixels copied out of the
    /// shared buffer).
    func render() async throws -> CGImage {
        try await withCheckedThrowingContinuation { continuation in
            do {
                let commandBuffers = try encodeFrame()
                // A queue runs its command buffers in order, so the last one
                // completing means every band is done. Command buffers are safe
                // to query from the completion thread once they have finished.
                let bands = UncheckedSendable(commandBuffers)
                commandBuffers.last!.addCompletedHandler { [self] _ in
                    continuation.resume(with: Result {
                        try bands.value.forEach(checkStatus)
                        return makeImage()
                    })
                }
                commandBuffers.forEach { $0.commit() }
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    /// Synchronous variant for tools/tests: blocks until the GPU finishes.
    func renderBlocking() throws -> CGImage {
        let commandBuffers = try encodeFrame()
        commandBuffers.forEach { $0.commit() }
        commandBuffers.last!.waitUntilCompleted()
        try commandBuffers.forEach(checkStatus)
        return makeImage()
    }

    /// Direct read access to the BGRA8 pixels (rows are `bytesPerRow` apart).
    /// Only valid between renders.
    func withPixels<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        try body(UnsafeRawBufferPointer(start: outputBuffer.contents(), count: outputBuffer.length))
    }

    // MARK: Private

    /// Encodes one frame as one command buffer per band; the caller commits.
    private func encodeFrame() throws -> [MTLCommandBuffer] {
        // Write the uniforms straight into shared memory. Safe because renders are
        // serialized, so the GPU isn't reading this buffer right now.
        uniformBuffer.contents().storeBytes(of: uniforms, as: GolfSceneUniforms.self)

        let tg = threadgroupSize
        let nonUniform = device.supportsFamily(.apple4) || device.supportsFamily(.mac2)

        return try (0..<bandCount).map { band in
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw GolfBallRendererError.commandBufferFailed(nil)
            }
            commandBuffer.label = "GolfScene Band \(band)"

            encoder.setComputePipelineState(pipeline)
            encoder.setTexture(outputTexture, index: 0)
            encoder.setBuffer(uniformBuffer, offset: 0, index: 0)
            encoder.setBuffer(tileOriginBuffer, offset: band * Self.tileOriginStride, index: 1)

            // 4. Dispatch one band: grid = full width x band height (last band clipped).
            let rows = min(bandHeight, height - band * bandHeight)
            let grid = MTLSize(width: width, height: rows, depth: 1)
            if nonUniform {
                // Non-uniform threadgroups: exact grid, edge groups clipped by Metal.
                encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
            } else {
                // Round up; out-of-range threads exit via the kernel's bounds check.
                let groups = MTLSize(width:  (grid.width  + tg.width  - 1) / tg.width,
                                     height: (grid.height + tg.height - 1) / tg.height,
                                     depth:  1)
                encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
            }
            encoder.endEncoding()
            return commandBuffer
        }
    }

    private func checkStatus(_ commandBuffer: MTLCommandBuffer) throws {
        guard commandBuffer.status == .completed else {
            throw GolfBallRendererError.commandBufferFailed(commandBuffer.error)
        }
    }

    /// 5. CPU readback: the GPU wrote directly into `outputBuffer`'s memory.
    private func makeImage() -> CGImage {
        let data = Data(bytes: outputBuffer.contents(), count: bytesPerRow * height)
        let provider = CGDataProvider(data: data as CFData)!
        let bitmapInfo = CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Little.rawValue |
                                                CGImageAlphaInfo.noneSkipFirst.rawValue)   // BGRA8
        return CGImage(width: width, height: height,
                       bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: bytesPerRow,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!,
                       bitmapInfo: bitmapInfo, provider: provider,
                       decode: nil, shouldInterpolate: true, intent: .defaultIntent)!
    }
}

// MARK: - SwiftUI view ----------------------------------------------------------

/// Shows the rendered address position. The scene is static, so each camera
/// choice is rendered once, read back once and displayed as a CGImage.
struct GolfBallView: View {
    var width = 1600
    var height = 1200

    @State private var view: AddressView = .broadcast
    @State private var renderer: GolfBallRenderer?
    @State private var image: CGImage?
    @State private var isRendering = false
    @State private var errorText: String?

    var body: some View {
        ZStack(alignment: .top) {
            Color.black
            if let image {
                Image(decorative: image, scale: 1).resizable().scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let errorText {
                Text(errorText).foregroundStyle(.red)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            Picker("View", selection: $view) {
                ForEach(AddressView.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            .padding(12)
            .disabled(isRendering)       // one render in flight: the output buffer is reused
        }
        .task(id: view) { await renderCurrentView() }
    }

    private func renderCurrentView() async {
        isRendering = true
        defer { isRendering = false }
        do {
            let renderer = try renderer ?? GolfBallRenderer(width: width, height: height)
            self.renderer = renderer
            renderer.uniforms.cameraPosition = view.cameraPosition
            renderer.uniforms.cameraTarget = view.cameraTarget
            renderer.uniforms.focusDistance = view.focusDistance
            image = try await renderer.render()
        } catch {
            errorText = "Metal error: \(error)"
        }
    }
}
