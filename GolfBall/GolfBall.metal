//
//  GolfBall.metal
//
//  Compute-kernel renderer: Sunday at the Masters. A right-handed player in
//  red-and-black Sunday attire addresses a teed ball with a T100-style 4-iron,
//  with Augusta-style pines, azaleas and a low-detail gallery behind.
//  The scene is static: it is rendered once and read back once.
//
//  Technique overview
//  ------------------
//  * One GPU thread == one output pixel. NxN samples per pixel, each with
//    thin-lens jitter, give anti-aliasing and depth of field in one pass,
//    so the focus stays on the player, the club and the ball.
//  * Units are centimetres. The parts use real proportions:
//      ball    Ø 4.267 cm, 392 dimples, teed low for an iron
//      4-iron  2025 T100-style dual-cavity back, 23° loft, 61.5° lie, 97 cm to the grip cap
//      player  185 cm. The address pose is solved offline with 2-bone IK
//              (see "Golfer pose"): straight-ish lead arm, soft trail arm,
//              32° hip hinge, hands slightly ahead of the ball.
//  * Right-handed setup: target line = +X, the player stands on the -Z side
//    facing +Z, so the target is on his left.
//  * Geometry is one sphere-traced signed distance field (SDF). Bounding
//    volumes skip whole groups (hero props, player, gallery, trees) that are
//    far from the sample point.
//  * Markings (the "Titleist"-style script, the T100 badge and the club
//    number) come from a tiny single-stroke vector font evaluated as a 2D SDF
//    on the back of the head.
//
//  ---------------------------------------------------------------------------
//  THREAD / THREADGROUP LAYOUT (host side, Swift)
//  ---------------------------------------------------------------------------
//  The grid is 2D. Each dispatch covers one horizontal band (tile) of the
//  image, and the kernel adds the band's origin:
//      pixel = tileOrigin + thread_position_in_grid
//  The bands keep each GPU submission short. macOS can kill one long-running
//  compute submission on the display GPU, and the band split avoids that.
//
//  Threadgroup = one SIMD-group wide (threadExecutionWidth, 32 on Apple
//  GPUs) and as tall as the pipeline allows. A SIMD-group then covers a
//  horizontal run of neighbouring pixels. Those rays mostly hit the same
//  object, which keeps divergence in the sphere-tracing loops low.
//
//      let w  = pipeline.threadExecutionWidth
//      let h  = pipeline.maxTotalThreadsPerThreadgroup / w
//      let tg = MTLSize(width: w, height: h, depth: 1)
//      // per band: grid = (W, bandHeight)
//      encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)          // non-uniform
//      // or, without non-uniform threadgroup support:
//      encoder.dispatchThreadgroups(ceil(grid / tg), threadsPerThreadgroup: tg)
//
//  * No threadgroup memory or barriers are used, so out-of-range threads can
//    return early. Add a barrier and that must become a guarded write.
//  * Set `debugTiles = 1` to tint each threadgroup tile on screen.
//
//  ---------------------------------------------------------------------------
//  Matching Swift structs:
//
//      struct GolfSceneUniforms {                // 64 bytes
//          var cameraPosition: SIMD4<Float>      // xyz cm, w = lens aperture radius (cm)
//          var cameraTarget:   SIMD4<Float>      // xyz cm, w = focal length
//          var focusDistance:  Float             // cm along the view axis
//          var samplesPerAxis: UInt32
//          var dimpleCount:    UInt32
//          var debugTiles:     UInt32
//          var padding:        SIMD4<UInt32>
//      }
//      tileOrigin: SIMD2<UInt32> at buffer(1)    // per-band origin, from a shared MTLBuffer
//

#include <metal_stdlib>
using namespace metal;

// MARK: - Shared types ----------------------------------------------------------

struct GolfSceneUniforms {
    float4 cameraPosition;   // xyz, w = aperture radius
    float4 cameraTarget;     // xyz, w = focal length
    float  focusDistance;
    uint   samplesPerAxis;
    uint   dimpleCount;
    uint   debugTiles;
    uint4  padding;
};

// MARK: - Materials -------------------------------------------------------------

enum : int {
    MAT_NONE = 0,
    MAT_BALL, MAT_TEE, MAT_HEAD, MAT_FERRULE, MAT_SHAFT, MAT_GRIP,
    MAT_SHIRT, MAT_PANTS, MAT_BELT, MAT_SKIN, MAT_GLOVE, MAT_CAP, MAT_SHOE,
    MAT_FAN_SHIRT, MAT_FAN_PANTS, MAT_FAN_SKIN, MAT_FAN_HAT,
    MAT_BAG, MAT_HEADCOVER, MAT_BAG_IRON, MAT_TOWEL, MAT_MARKER, MAT_BOARD,
    MAT_ROPE, MAT_TRUNK, MAT_CANOPY, MAT_AZALEA, MAT_TERRAIN
};

// MARK: - Hero props (cm) -------------------------------------------------------

constant float  kPHI          = 1.61803398875f;

constant float  kBallRadius   = 2.1335f;                  // 42.67 mm
constant float  kBallCenterY  = 3.6587f;                  // tee rim + sqrt(R^2 - cupR^2)
constant float3 kBallCenter   = float3(0.0f, kBallCenterY, 0.0f);

constant float  kTeeTopY      = 1.60f;
constant float  kTeeCupRadius = 0.56f;
constant float  kTeeHeadDepth = 0.75f;
constant float  kTeeShankR    = 0.25f;

// 2025 T100-style 4-iron. Toe points away from the player (+Z), heel toward him.
constant float  kLoftTan      = 0.42447f;                 // tan 23° (2025 T100 4-iron)
constant float  kLoftCos      = 0.92050f;                 // cos 23°
constant float  kLeadEdgeX    = -1.1076f;                 // face sits 3 mm behind the ball
constant float  kToeZ         =  4.20f;
constant float  kHeelZ        = -3.40f;
constant float  kToeHeight    =  4.70f;
constant float  kHeelHeight   =  3.30f;
constant float  kSoleLift     =  0.04f;
constant float3 kHoselBase    = float3(-1.80f, 0.60f, -2.40f);
constant float  kHoselLength  = 7.0f;
constant float  kFerruleLength = 1.3f;
constant float  kGripLength   = 27.0f;

// MARK: - Golfer pose (cm) ------------------------------------------------------
// Solved offline (tools/pose2.py) with 2-bone IK for a 185 cm player: joint-centre
// segments upper arm 31, forearm 26.5, thigh 47, shin 46; 37° forward tilt like
// Tiger's mid-iron setup, shoulders slightly open, lead arm ~17° and trail arm ~26°
// bent with both elbows pointing at the hips, holding the 4-iron at its 61.5° lie (2025 T100 4-iron spec) with slight forward
// shaft lean. Grip cap = hosel base + shaft direction * 97 cm.

constant float3 kShaftDir      = float3(0.100f, 0.874f, -0.475f);
constant float3 kGripEnd       = float3(7.852f, 85.422f, -48.455f);
constant float3 kLeadHand      = float3(7.305f, 80.613f, -45.843f);
constant float3 kTrailHand     = float3(6.608f, 74.492f, -42.520f);
constant float3 kAnkleLead     = float3(15.000f, 9.000f, -86.000f);
constant float3 kAnkleTrail    = float3(-29.000f, 9.000f, -86.000f);
constant float3 kKneeLead      = float3(17.263f, 53.290f, -73.784f);
constant float3 kKneeTrail     = float3(-19.117f, 52.813f, -76.062f);
constant float3 kHipLead       = float3(10.500f, 93.000f, -98.000f);
constant float3 kHipTrail      = float3(-8.500f, 93.000f, -98.000f);
constant float3 kPelvis        = float3(1.000f, 96.000f, -98.000f);
constant float3 kSpineUp       = float3(-0.138f, 0.791f, 0.596f);
constant float3 kSpineSide     = float3(0.990f, 0.110f, 0.083f);
constant float3 kSpineFwd      = float3(0.000f, -0.602f, 0.799f);
constant float3 kShoulderLead  = float3(9.211f, 142.878f, -61.999f);
constant float3 kShoulderTrail = float3(-21.547f, 126.573f, -65.620f);
constant float3 kNeck          = float3(-7.133f, 142.670f, -62.832f);
constant float3 kHead          = float3(-8.649f, 148.061f, -51.883f);
constant float3 kHeadFace      = float3(0.050f, -0.766f, 0.641f);   // ~50° down; the eyes do the rest
constant float3 kHeadUp        = float3(-0.120f, 0.636f, 0.762f);
constant float3 kElbowLead     = float3(8.523f, 112.092f, -58.427f);
constant float3 kElbowTrail    = float3(-7.591f, 99.367f, -60.516f);
constant float3 kWristLead     = float3(7.527f, 87.869f, -47.726f);
constant float3 kWristTrail    = float3(3.285f, 80.638f, -45.246f);

// Feet: lead foot flared ~25° toward the target, trail foot ~8°.
constant float3 kFootDirLead   = float3( 0.423f, 0.0f, 0.906f);
constant float3 kFootDirTrail  = float3(-0.139f, 0.0f, 0.990f);

// MARK: - Staff bag (cm) ---------------------------------------------------------
// Tour staff bag set down behind the player's trail side, front panel toward
// the broadcast camera.

constant float3 kBagBase   = float3(-150.0f, 0.0f, -235.0f);
constant float3 kBagUp     = float3(0.050f, 0.991f, -0.124f);   // slight lean back
constant float  kBagHeight = 92.0f;
constant float  kBagRadius = 14.5f;

// MARK: - Course & gallery (cm) -------------------------------------------------

constant float kRopeZ       = -1080.0f;
constant float kFanRowZ0    = -1220.0f;
constant float kFanRowDZ    = 92.0f;
constant int   kFanRows     = 6;
constant float kFanCellX    = 72.0f;
constant float kTreeZ0      = -2250.0f;
constant float kTreeRowDZ   = 620.0f;
constant int   kTreeRows    = 4;
constant float kTreeCellX   = 430.0f;
constant float3 kTeeMarkerA = float3(62.0f, 0.0f, 235.0f);      // tee markers, 62 cm ahead of the ball
constant float3 kTeeMarkerB = float3(62.0f, 0.0f, -330.0f);
constant float3 kBoardMin   = float3(200.0f, 330.0f, -1965.0f);  // hand-operated leaderboard
constant float3 kBoardMax   = float3(1000.0f, 750.0f, -1935.0f);

// Lighting: final round on the back nine, ~5 pm in April. Sun about 30° up,
// behind the camera and toward the target, warm and raking.
// kSunColor is sun irradiance / pi, so Lambert diffuse = albedo * kSunColor * NoL.
constant float3 kSunDir      = float3(0.45f, 0.50f, 0.74f);    // normalized in code
constant float3 kSunColor    = float3(1.00f, 0.86f, 0.68f) * 3.2f;
constant float3 kSkyTop      = float3(0.26f, 0.46f, 0.84f);
constant float3 kSkyHorizon  = float3(0.76f, 0.83f, 0.92f);
constant float3 kGroundBounce = float3(0.10f, 0.19f, 0.06f);   // light bounced off the turf
constant float3 kScatterTint = float3(1.00f, 0.40f, 0.25f);    // subsurface colour at the terminator

// Dimples
constant float kDimpleRadiusScale = 0.47f;
constant float kDimpleSlope       = 0.55f;
constant float kDimpleCavityAO    = 0.72f;

// Ray marching
constant int   kMaxSteps    = 300;
constant float kMaxDistance = 9000.0f;

// MARK: - Single-stroke font (club markings) -----------------------------------
// Generated by glyphs.py: single-stroke glyphs, cap height = 1.
// Index: 0='T', 1='1', 2='0', 3='4', 4='S', 5='i', 6='t', 7='l', 8='e', 9='s'
constant float4 kGlyphSegments[56] = {
    float4(0.000f, 1.000f, 0.620f, 1.000f),
    float4(0.310f, 1.000f, 0.310f, 0.000f),
    float4(0.060f, 0.780f, 0.280f, 1.000f),
    float4(0.280f, 1.000f, 0.280f, 0.000f),
    float4(0.560f, 0.500f, 0.535f, 0.717f),
    float4(0.535f, 0.717f, 0.466f, 0.891f),
    float4(0.466f, 0.891f, 0.366f, 0.987f),
    float4(0.366f, 0.987f, 0.254f, 0.987f),
    float4(0.254f, 0.987f, 0.154f, 0.891f),
    float4(0.154f, 0.891f, 0.085f, 0.717f),
    float4(0.085f, 0.717f, 0.060f, 0.500f),
    float4(0.060f, 0.500f, 0.085f, 0.283f),
    float4(0.085f, 0.283f, 0.154f, 0.109f),
    float4(0.154f, 0.109f, 0.254f, 0.013f),
    float4(0.254f, 0.013f, 0.366f, 0.013f),
    float4(0.366f, 0.013f, 0.466f, 0.109f),
    float4(0.466f, 0.109f, 0.535f, 0.283f),
    float4(0.535f, 0.283f, 0.560f, 0.500f),
    float4(0.460f, 0.000f, 0.460f, 1.000f),
    float4(0.460f, 1.000f, 0.000f, 0.320f),
    float4(0.000f, 0.320f, 0.620f, 0.320f),
    float4(0.000f, 0.880f, 0.180f, 1.000f),
    float4(0.180f, 1.000f, 0.550f, 0.980f),
    float4(0.550f, 0.980f, 0.800f, 1.050f),
    float4(0.450f, 0.980f, 0.300f, 0.450f),
    float4(0.300f, 0.450f, 0.220f, 0.120f),
    float4(0.220f, 0.120f, 0.100f, 0.000f),
    float4(0.100f, 0.000f, 0.000f, 0.080f),
    float4(0.000f, 0.000f, 0.080f, 0.060f),
    float4(0.080f, 0.060f, 0.120f, 0.450f),
    float4(0.150f, 0.620f, 0.160f, 0.660f),
    float4(0.060f, 0.060f, 0.220f, 0.120f),
    float4(0.040f, 0.020f, 0.140f, 0.780f),
    float4(0.000f, 0.480f, 0.300f, 0.500f),
    float4(0.100f, 0.020f, 0.300f, 0.120f),
    float4(0.000f, 0.040f, 0.120f, 0.300f),
    float4(0.120f, 0.300f, 0.200f, 0.820f),
    float4(0.200f, 0.820f, 0.160f, 0.950f),
    float4(0.160f, 0.950f, 0.080f, 0.800f),
    float4(0.080f, 0.800f, 0.060f, 0.200f),
    float4(0.060f, 0.200f, 0.120f, 0.020f),
    float4(0.120f, 0.020f, 0.240f, 0.100f),
    float4(0.040f, 0.200f, 0.240f, 0.300f),
    float4(0.240f, 0.300f, 0.300f, 0.420f),
    float4(0.300f, 0.420f, 0.220f, 0.480f),
    float4(0.220f, 0.480f, 0.100f, 0.400f),
    float4(0.100f, 0.400f, 0.040f, 0.180f),
    float4(0.040f, 0.180f, 0.120f, 0.020f),
    float4(0.120f, 0.020f, 0.360f, 0.100f),
    float4(0.020f, 0.100f, 0.140f, 0.020f),
    float4(0.140f, 0.020f, 0.280f, 0.060f),
    float4(0.280f, 0.060f, 0.260f, 0.200f),
    float4(0.260f, 0.200f, 0.120f, 0.300f),
    float4(0.120f, 0.300f, 0.120f, 0.420f),
    float4(0.120f, 0.420f, 0.240f, 0.480f),
    float4(0.240f, 0.480f, 0.320f, 0.420f),
};
constant float4 kGlyphInfo[10] = {   // (first segment, segment count, advance, 0)
    float4(0.0f, 2.0f, 0.62f, 0.0f),   // 'T'
    float4(2.0f, 2.0f, 0.42f, 0.0f),   // '1'
    float4(4.0f, 14.0f, 0.62f, 0.0f),   // '0'
    float4(18.0f, 3.0f, 0.62f, 0.0f),   // '4'
    float4(21.0f, 7.0f, 0.80f, 0.0f),   // 'S'
    float4(28.0f, 4.0f, 0.22f, 0.0f),   // 'i'
    float4(32.0f, 3.0f, 0.30f, 0.0f),   // 't'
    float4(35.0f, 7.0f, 0.24f, 0.0f),   // 'l'
    float4(42.0f, 7.0f, 0.36f, 0.0f),   // 'e'
    float4(49.0f, 7.0f, 0.34f, 0.0f),   // 's'
};
constant uchar kTextT100[4] = { 0, 1, 2, 2 };
constant uchar kTextTitleist[8] = { 4, 5, 6, 7, 8, 5, 9, 6 };
constant uchar kTextFour[1] = { 3 };

// MARK: - Math helpers ----------------------------------------------------------

static float3x3 rotationAxisAngle(float3 a, float angle)
{
    const float s = sin(angle), c = cos(angle), ic = 1.0f - c;
    return float3x3(
        float3(a.x * a.x * ic + c,       a.y * a.x * ic + a.z * s, a.z * a.x * ic - a.y * s),
        float3(a.x * a.y * ic - a.z * s, a.y * a.y * ic + c,       a.z * a.y * ic + a.x * s),
        float3(a.x * a.z * ic + a.y * s, a.y * a.z * ic - a.x * s, a.z * a.z * ic + c));
}

static float smin(float a, float b, float k)
{
    const float h = saturate(0.5f + 0.5f * (b - a) / k);
    return mix(b, a, h) - k * h * (1.0f - h);
}

// Union that carries the material of the closer surface.
static float2 opU(float2 a, float2 b) { return (a.x < b.x) ? a : b; }

static float hash12(float2 p)
{
    float3 p3 = fract(float3(p.xyx) * 0.1031f);
    p3 += dot(p3, p3.yzx + 33.33f);
    return fract((p3.x + p3.y) * p3.z);
}

static float valueNoise(float2 p)
{
    const float2 i = floor(p);
    const float2 f = fract(p);
    const float2 u = f * f * (3.0f - 2.0f * f);
    return mix(mix(hash12(i),                hash12(i + float2(1, 0)), u.x),
               mix(hash12(i + float2(0, 1)), hash12(i + float2(1, 1)), u.x), u.y);
}

static float fbm(float2 p)
{
    float sum = 0.0f, amp = 0.5f;
    for (int i = 0; i < 4; ++i) {
        sum += amp * valueNoise(p);
        p    = float2x2(1.6f, 1.2f, -1.2f, 1.6f) * p;
        amp *= 0.5f;
    }
    return sum;
}

// MARK: - SDF primitives (after Inigo Quilez) ----------------------------------

static float sdCapsule(float3 p, float3 a, float3 b, float r)
{
    const float3 pa = p - a, ba = b - a;
    return length(pa - ba * saturate(dot(pa, ba) / dot(ba, ba))) - r;
}

static float sdRoundCone(float3 p, float3 a, float3 b, float r1, float r2)
{
    const float3 ba  = b - a;
    const float  l2  = dot(ba, ba);
    const float  rr  = r1 - r2;
    const float  a2  = l2 - rr * rr;
    const float  il2 = 1.0f / l2;
    const float3 pa  = p - a;
    const float  y   = dot(pa, ba);
    const float  z   = y - l2;
    const float3 xv  = pa * l2 - ba * y;
    const float  x2  = dot(xv, xv);
    const float  y2  = y * y * l2;
    const float  z2  = z * z * l2;
    const float  k   = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k) return sqrt(x2 + z2) * il2 - r2;
    if (sign(y) * a2 * y2 < k) return sqrt(x2 + y2) * il2 - r1;
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

static float sdCappedCone(float3 p, float3 a, float3 b, float ra, float rb)
{
    const float rba  = rb - ra;
    const float baba = dot(b - a, b - a);
    const float papa = dot(p - a, p - a);
    const float paba = dot(p - a, b - a) / baba;
    const float x    = sqrt(max(0.0f, papa - paba * paba * baba));
    const float cax  = max(0.0f, x - ((paba < 0.5f) ? ra : rb));
    const float cay  = abs(paba - 0.5f) - 0.5f;
    const float k    = rba * rba + baba;
    const float f    = saturate((rba * (x - ra) + paba * baba) / k);
    const float cbx  = x - ra - f * rba;
    const float cby  = paba - f;
    const float s    = (cbx < 0.0f && cay < 0.0f) ? -1.0f : 1.0f;
    return s * sqrt(min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba));
}

static float sdEllipsoid(float3 p, float3 r)
{
    const float k0 = length(p / r);
    const float k1 = max(length(p / (r * r)), 1e-6f);
    return k0 * (k0 - 1.0f) / k1;
}

static float sdRoundBox(float3 p, float3 b, float r)
{
    const float3 q = abs(p) - b + r;
    return length(max(q, 0.0f)) + min(max(q.x, max(q.y, q.z)), 0.0f) - r;
}

static float sdBox(float3 p, float3 b)
{
    const float3 q = abs(p) - b;
    return length(max(q, 0.0f)) + min(max(q.x, max(q.y, q.z)), 0.0f);
}

static float sdRoundBox2(float2 p, float2 b, float4 r)
{
    r.xy = (p.x > 0.0f) ? r.xy : r.zw;
    r.x  = (p.y > 0.0f) ? r.x  : r.y;
    const float2 q = abs(p) - b + r.x;
    return min(max(q.x, q.y), 0.0f) + length(max(q, 0.0f)) - r.x;
}

static float sdSegment2(float2 p, float2 a, float2 b)
{
    const float2 pa = p - a, ba = b - a;
    return length(pa - ba * saturate(dot(pa, ba) / dot(ba, ba)));
}

// MARK: - Vector text -----------------------------------------------------------

// Distance (in cap-height units) to a single-stroke string laid out from x = 0.
// `slant` shears the glyphs for an italic/script look.
static float sdText(float2 p, constant uchar* text, int count, float tracking, float slant)
{
    p.x -= p.y * slant;
    float d = 1e9f;
    float pen = 0.0f;
    for (int c = 0; c < count; ++c) {
        const float4 info = kGlyphInfo[text[c]];
        const float2 q    = p - float2(pen, 0.0f);
        if (q.x > -0.3f && q.x < info.z + 0.3f) {            // only nearby glyphs
            const int first = int(info.x), n = int(info.y);
            for (int s = 0; s < n; ++s) {
                const float4 seg = kGlyphSegments[first + s];
                d = min(d, sdSegment2(q, seg.xy, seg.zw));
            }
        }
        pen += info.z + tracking;
    }
    return d;
}

static float textWidth(constant uchar* text, int count, float tracking)
{
    float w = 0.0f;
    for (int c = 0; c < count; ++c) w += kGlyphInfo[text[c]].z + tracking;
    return w - tracking;
}

// The Swoosh, traced from the official vector outline (Wikimedia Logo_NIKE.svg)
// by tools/swoosh.py: Bézier path flattened to a polygon in unit space (width 2,
// centred, y up).
constant float2 kSwoosh[36] = {
    float2(-0.7794f, -0.3561f), float2(-0.8321f, -0.3498f), float2(-0.8785f, -0.3353f), float2(-0.9186f, -0.3127f),
    float2(-0.9501f, -0.2837f), float2(-0.9739f, -0.2513f), float2(-0.9893f, -0.2174f), float2(-0.9980f, -0.1790f),
    float2(-0.9990f, -0.1220f), float2(-0.9880f, -0.0594f), float2(-0.9719f, -0.0094f), float2(-0.9505f, 0.0400f),
    float2(-0.9172f, 0.1005f), float2(-0.8859f, 0.1480f), float2(-0.8338f, 0.2163f), float2(-0.7110f, 0.3564f),
    float2(-0.7506f, 0.2800f), float2(-0.7782f, 0.1990f), float2(-0.7851f, 0.1618f), float2(-0.7876f, 0.1268f),
    float2(-0.7856f, 0.0943f), float2(-0.7791f, 0.0642f), float2(-0.7682f, 0.0369f), float2(-0.7500f, 0.0086f),
    float2(-0.7265f, -0.0158f), float2(-0.7049f, -0.0314f), float2(-0.6829f, -0.0424f), float2(-0.6480f, -0.0531f),
    float2(-0.5790f, -0.0594f), float2(-0.5150f, -0.0541f), float2(-0.4622f, -0.0441f), float2(1.0000f, 0.3443f),
    float2(-0.4955f, -0.2949f), float2(-0.5735f, -0.3225f), float2(-0.6538f, -0.3437f), float2(-0.7269f, -0.3547f),
};

// Exact signed distance to the Swoosh polygon (after Inigo Quilez's sdPolygon).
static float sdSwoosh(float2 p)
{
    if (abs(p.x) > 1.1f || abs(p.y) > 0.5f) return 1.0f;
    const int n = 36;
    float d = dot(p - kSwoosh[0], p - kSwoosh[0]), s = 1.0f;
    for (int i = 0, j = n - 1; i < n; j = i, ++i) {
        const float2 e = kSwoosh[j] - kSwoosh[i], w = p - kSwoosh[i];
        const float2 b = w - e * saturate(dot(w, e) / dot(e, e));
        d = min(d, dot(b, b));
        const bool3 c = bool3(p.y >= kSwoosh[i].y, p.y < kSwoosh[j].y, e.x * w.y > e.y * w.x);
        if (all(c) || all(!c)) s = -s;
    }
    return s * sqrt(d);
}

// MARK: - Terrain ---------------------------------------------------------------

// Flat tee box, then spectator mounds rising behind the rope and a gentle
// climb into the pines.
static float terrainHeight(float z)
{
    return 170.0f * smoothstep(-1120.0f, -1900.0f, z)
         + 240.0f * smoothstep(-1900.0f, -4400.0f, z);
}

// MARK: - Hero props: ball, tee, club --------------------------------------------

static float sdBall(float3 p) { return length(p - kBallCenter) - kBallRadius; }

static float sdTee(float3 p)
{
    const float3 headBottom = float3(0.0f, kTeeTopY - kTeeHeadDepth, 0.0f);
    const float shank = sdCappedCone(p, float3(0.0f, -3.0f, 0.0f), headBottom,
                                     kTeeShankR * 0.85f, kTeeShankR);
    const float head  = sdCappedCone(p, headBottom, float3(0.0f, kTeeTopY, 0.0f),
                                     kTeeShankR, kTeeCupRadius);
    return max(smin(shank, head, 0.12f), -(sdBall(p) + 0.005f));
}

// T100 (2025) back geometry in blade coordinates (u = heel->toe, y = height).
// Outline: compact length, rounded high toe, straight top line falling to the heel.
static float bladeProfile(float u, float y)
{
    const float  zc    = 0.5f * (kToeZ + kHeelZ);
    const float2 halfs = float2(0.5f * (kToeZ - kHeelZ), 0.5f * kToeHeight);
    float d = sdRoundBox2(float2(u - zc, y - halfs.y), halfs,
                          float4(1.65f, 0.85f, 0.45f, 0.55f));     // toe top/bottom, heel top/bottom
    const float2 nTop = normalize(float2(-(kToeHeight - kHeelHeight), kToeZ - kHeelZ));
    return max(d, dot(float2(u - kHeelZ, y - kHeelHeight), nTop));
}

// Top edge of the lower muscle: the angled T-series cavity line, low heel to high toe.
static float muscleLine(float u)
{
    return mix(1.30f, 1.85f, saturate((u - kHeelZ) / (kToeZ - kHeelZ)));
}

constant float kFloorThickness = 0.34f;                   // cavity floor behind the face
constant float kFrameWidth     = 0.48f;                   // top/toe perimeter frame
constant float kBadgeLift      = 0.07f;
constant float2 kBadgeCenter   = float2(0.55f, 2.50f);
constant float2 kBadgeHalf     = float2(2.15f, 0.46f);

constant float kEdgeRadius     = 0.05f;                   // forged edges are softly broken

// Lofted slab between the face (s = 0) and a back plane at depth t, shrunk by the
// edge radius (added back at the end of sdIronBlade to round every edge).
static float bladeSlab(float s, float t) { return (max(s, -s - t) + kEdgeRadius) * kLoftCos; }

// Forged T100-style head assembled from flat machined layers so every ledge stays
// a crisp line and the SDF stays Lipschitz: face plate + perimeter frame + lower
// muscle, a raised badge, the dual-cavity undercut behind the muscle lip, the
// long-iron muscle channel between the tungsten weights and a variable-bounce sole.
static float sdIronBlade(float3 p)
{
    const float3 q  = p - float3(kLeadEdgeX, kSoleLift, 0.0f);
    const float  u  = q.z;
    const float  uN = saturate((u - kHeelZ) / (kToeZ - kHeelZ));
    const float  s  = q.x + q.y * kLoftTan;                    // 0 on the face plane
    const float  outline = bladeProfile(u, q.y) + kEdgeRadius;
    const float  ml = muscleLine(u);

    const float plate  = bladeSlab(s, kFloorThickness);
    // Perimeter frame: thin top line that thickens down the heel and toe into the
    // tungsten pads, so the toe silhouette flows into the muscle without a step.
    const float pad    = smoothstep(ml + 1.8f, ml - 0.2f, q.y);
    const float frameT = mix(0.50f, 0.60f, uN) + 0.85f * pad;
    float frame = max(bladeSlab(s, frameT), -outline - kFrameWidth);
    frame = max(frame, (-s - frameT + 0.22f + outline) * 0.6f);   // 45° back bevel slims the top line
    const float muscle = max(bladeSlab(s, mix(2.10f, 1.05f, smoothstep(0.0f, ml, q.y))),
                             q.y - ml + kEdgeRadius);
    const float badge  = max(bladeSlab(s, kFloorThickness + kBadgeLift),
                             sdRoundBox2(float2(u, q.y) - kBadgeCenter, kBadgeHalf, float4(0.35f)));
    float d = max(outline, min(min(smin(plate, frame, 0.05f), muscle), badge));

    // Dual cavity: a slot behind the muscle lip, open to the upper cavity.
    const float3 slot = float3(u - 0.35f, q.y - (ml - 0.25f), s + 0.72f);
    d = max(d, -sdBox(slot, float3(2.75f, 0.32f, 0.36f)) * kLoftCos);

    // Muscle channel milled across the back of the sole bar (3- and 4-irons).
    const float3 chan = float3(u - 0.40f, q.y - 0.38f, s + 2.6f);
    d = max(d, -(sdBox(chan, float3(2.05f, 0.14f, 0.78f)) - 0.03f));

    // Variable-bounce sole: camber toward heel and toe plus a relieved trailing
    // edge that grows toward the heel.
    const float cu     = (u - 0.5f * (kToeZ + kHeelZ)) / (0.5f * (kToeZ - kHeelZ));
    const float camber = 0.10f * cu * cu;
    d = max(d, camber - q.y + kEdgeRadius);
    const float relief = mix(1.45f, 1.70f, uN);
    d = max(d, ((-s - relief) * 0.32f - q.y * 0.55f) * (kLoftCos / 0.6364f) + kEdgeRadius);
    return d - kEdgeRadius;
}

static float3 hoselTop()   { return kHoselBase + kShaftDir * kHoselLength; }
static float3 ferruleTop() { return hoselTop() + kShaftDir * kFerruleLength; }
static float3 gripStart()  { return kGripEnd - kShaftDir * kGripLength; }

static float2 mapHero(float3 p)
{
    float2 res = float2(sdBall(p), float(MAT_BALL));
    res = opU(res, float2(sdTee(p), float(MAT_TEE)));

    const float hosel = sdCappedCone(p, kHoselBase, hoselTop(), 0.78f, 0.58f);
    res = opU(res, float2(smin(sdIronBlade(p), hosel, 0.40f), float(MAT_HEAD)));
    return res;
}

static float2 mapShaft(float3 p)
{
    float2 res = float2(sdCappedCone(p, hoselTop(), ferruleTop(), 0.58f, 0.46f), float(MAT_FERRULE));
    res = opU(res, float2(sdCappedCone(p, ferruleTop(), gripStart(), 0.44f, 0.60f), float(MAT_SHAFT)));
    res = opU(res, float2(sdRoundCone(p, gripStart(), kGripEnd - kShaftDir * 0.6f, 0.95f, 1.15f), float(MAT_GRIP)));
    return res;
}

// MARK: - Golfer ---------------------------------------------------------------
// Sculpted from anatomical forms (pelvis, rib cage, pecs, lats, deltoids,
// biceps/triceps, forearm flexors, skull/jaw/brow/cheekbones). Clothing is a
// separate, looser layer with folds where fabric really creases at address:
//   shirt    - compression folds across the abdomen from the hip hinge,
//              blousing at the waistband, sleeve folds toward the hem
//   trousers - compression folds behind the knee, diagonal pull from the
//              crotch, and the "break" stacked on the shoe
// Folds are displacements of the cloth distance field. Their slope is bounded,
// so the cloth SDF is scaled by kClothLipschitz to keep sphere tracing safe.

constant float kClothLipschitz = 0.7f;

static float3 toSpine(float3 p)
{
    const float3 d = p - kPelvis;
    return float3(dot(d, kSpineSide), dot(d, kSpineUp), dot(d, kSpineFwd));
}

static float smax(float a, float b, float k) { return -smin(-a, -b, k); }

// Ellipsoid muscle at fraction t along bone a->b. r = (across, along, depth);
// `depthHint` orients the depth axis (e.g. the front of the arm).
static float sdMuscle(float3 p, float3 a, float3 b, float t, float3 r, float3 depthHint)
{
    const float3 u = normalize(b - a);
    const float3 w = normalize(depthHint - u * dot(depthHint, u));
    const float3 v = cross(u, w);
    const float3 d = p - mix(a, b, t);
    return sdEllipsoid(float3(dot(d, v), dot(d, u), dot(d, w)), r);
}

// Cloth fold cross-section: rounded crests, sharp creases, roughly zero-mean.
static float fold(float x) { return abs(sin(x)) - 0.64f; }

// Shirt torso (spine frame): anatomy under a loose polo, with folds.
static float sdShirtTorso(float3 t)
{
    const float3 m = float3(abs(t.x), t.y, t.z);                                  // mirrored
    // Abdomen as a straight-sided column (x squashed into an oval; the squash only
    // shrinks distances, so the SDF stays a safe lower bound): no pinched waist.
    float d = sdRoundCone(float3(t.x * 0.68f, t.y, t.z), float3(0.0f, 3.0f, -0.3f), float3(0.0f, 24.0f, 0.6f), 10.0f, 10.4f);
    d = smin(d, sdEllipsoid(t - float3(0.0f, 33.0f, 0.8f), float3(16.0f, 15.5f, 9.6f)), 6.0f);      // rib cage
    d = smin(d, sdEllipsoid(m - float3(8.0f, 34.0f, 7.0f), float3(8.5f, 6.5f, 3.4f)), 3.0f);        // pecs
    d = smin(d, sdEllipsoid(m - float3(11.5f, 30.0f, -1.5f), float3(5.5f, 12.0f, 6.8f)), 4.0f);     // lats: V taper
    d = smax(d, -t.z - 9.0f, 3.0f);                                                                 // straight back

    // Hip hinge compresses the front of the shirt into horizontal folds.
    const float front  = smoothstep(0.0f, 6.0f, t.z) * smoothstep(4.0f, 9.0f, t.y) * (1.0f - smoothstep(20.0f, 28.0f, t.y));
    const float warp   = 3.5f * valueNoise(t.xy * 0.10f + 3.0f);
    const float vary   = 0.4f + 0.6f * valueNoise(t.xy * 0.07f + 9.0f);   // folds come and go
    d -= 0.32f * front * vary * fold(t.y * 0.55f + warp);
    // Blousing where the shirt is tucked: vertical folds around the waistband.
    const float waist  = smoothstep(7.0f, 9.5f, t.y) * (1.0f - smoothstep(11.0f, 15.0f, t.y));
    d -= 0.30f * waist * vary * fold(atan2(t.x, t.z) * 9.0f + warp);
    return d;
}

// Trouser leg: loose tube hip -> knee -> shoe, with knee, crotch and break folds.
static float sdTrouserLeg(float3 p, float3 hip, float3 knee, float3 ankle, float side)
{
    float d = smin(sdRoundCone(p, hip, knee, 10.0f, 7.8f),                     // relaxed pleated fit
                   sdRoundCone(p, knee, ankle + float3(0, 2.0f, 0), 7.6f, 7.1f), 3.0f);

    const float warp = 1.8f * valueNoise(p.xy * 0.15f + side * 7.0f);
    // Behind the knee: horizontal compression folds, strongest at the back.
    const float3 kn   = normalize(p - knee + 1e-4f);
    const float  vary  = 0.4f + 0.6f * valueNoise(p.xy * 0.11f + side * 3.0f);
    const float  kneeW = exp(-(p.y - knee.y) * (p.y - knee.y) / 70.0f) * smoothstep(0.0f, -0.7f, kn.z);
    d -= 0.45f * kneeW * vary * fold((p.y - knee.y) * 0.8f + warp);
    // Flexed knees pull diagonal folds from the crotch toward the inner knee.
    const float thighW = smoothstep(knee.y + 6.0f, knee.y + 16.0f, p.y) * (1.0f - smoothstep(hip.y - 14.0f, hip.y - 4.0f, p.y));
    d -= 0.30f * thighW * vary * fold(dot(p, normalize(float3(side * 0.7f, 1.0f, 0.4f))) * 0.6f + warp);
    // The break: cloth stacks on the shoe.
    const float breakW = (1.0f - smoothstep(3.0f, 11.0f, p.y - ankle.y)) * smoothstep(-2.0f, 3.0f, p.z - ankle.z);
    d -= 0.45f * breakW * fold(p.y * 0.95f + warp * 1.5f);
    return d;
}

// Bare arm sculpted around its real bend. `ant` points into the inside of the
// elbow, so biceps, triceps, the elbow point and the forearm mass sit where they
// do on a real arm in any pose; the wrist flattens across the palm (`hand` is the
// grip centre the palm wraps).
static float sdArm(float3 p, float3 sh, float3 el, float3 wr, float3 hand)
{
    const float3 ua   = normalize(el - sh);
    const float3 bend = 0.5f * (sh + wr) - el;
    const float3 ant  = normalize(bend - ua * dot(bend, ua) + float3(0.0f, 0.0f, 1e-3f));
    const float3 side = normalize(cross(ua, ant));

    // Upper arm: humerus, biceps in front, triceps behind.
    float d = sdRoundCone(p, sh, el, 4.6f, 3.5f);
    d = smin(d, sdMuscle(p, sh, el, 0.58f, float3(3.6f, 8.0f, 3.8f),  ant), 2.5f);   // biceps
    d = smin(d, sdMuscle(p, sh, el, 0.38f, float3(4.0f, 9.5f, 3.8f), -ant), 2.5f);   // triceps
    // Elbow: olecranon behind, epicondyles either side, blended wide.
    d = smin(d, length(p - (el - ant * 2.4f)) - 1.6f, 1.8f);
    d = smin(d, min(length(p - (el + side * 2.8f)), length(p - (el - side * 2.8f))) - 1.4f, 1.8f);

    // Forearm: muscle mass below the elbow over a shaft that flattens into the wrist.
    float3 palm = wr - hand;
    palm = normalize(palm - kShaftDir * dot(palm, kShaftDir));
    float fore = sdRoundCone(p, el, wr, 3.7f, 2.3f);
    fore = smin(fore, sdMuscle(p, el, wr, 0.27f, float3(4.5f, 9.0f, 3.8f), ant), 2.5f);    // brachioradialis + flexors
    fore = smin(fore, sdMuscle(p, el, wr, 0.82f, float3(2.9f, 5.0f, 1.9f), palm), 1.5f);   // wrist, wide across
    return smin(d, fore, 3.0f);
}

// Deltoid: a cap over the shoulder joint, slightly lateral, tapering down the arm.
static float sdDeltoid(float3 p, float3 sh, float3 el)
{
    const float3 ua  = normalize(el - sh);
    float3 lat = sh - kNeck;
    lat = normalize(lat - ua * dot(lat, ua));
    return sdMuscle(p - lat * 1.2f, sh, el, 0.10f, float3(5.9f, 8.4f, 5.9f), cross(ua, lat));
}

// Short sleeve over the upper arm: loose cone, folds gathering toward the hem.
static float sdSleeve(float3 p, float3 sh, float3 el)
{
    const float3 hem = mix(sh, el, 0.52f);
    float d = sdRoundCone(p, sh, hem, 6.6f, 5.4f);
    const float3 u   = normalize(el - sh);
    const float  s   = dot(p - sh, u) / distance(sh, hem);
    const float3 r   = p - sh - u * dot(p - sh, u);
    const float  ang = atan2(dot(r, float3(0, 1, 0)), dot(r, cross(u, float3(0, 1, 0))));
    d -= 0.30f * smoothstep(0.3f, 0.7f, s) * (1.0f - smoothstep(0.78f, 0.95f, s)) * fold(ang * 4.0f + s * 2.5f);  // straight hem
    return d;
}

// Hand closed around the grip: palm on the wrist side, four fingers wrapping
// the grip, thumb laid down the shaft.
static float sdHand(float3 p, float3 center, float3 wrist)
{
    const float3 a     = kShaftDir;
    const float3 toW   = wrist - center;
    const float3 palmN = normalize(toW - a * dot(toW, a));             // grip -> palm direction
    const float palm = sdMuscle(p - palmN * 1.8f, center - a * 4.2f, center + a * 4.2f, 0.5f,
                                float3(3.4f, 4.6f, 1.9f), palmN);
    // Fingers: tori around the grip axis, index finger highest, pinky smallest.
    float fingers = 1e9f;
    for (int i = 0; i < 4; ++i) {
        const float3 c = center + a * (2.7f - 1.8f * float(i));
        const float3 q = p - c;
        const float  along = dot(q, a);
        const float  rad   = length(q - a * along);
        fingers = min(fingers, length(float2(rad - 1.95f, along)) - (0.88f - 0.07f * float(i)));
    }
    const float thumb = sdRoundCone(p, center + a * 2.5f + palmN * 1.4f, center - a * 1.5f - palmN * 0.6f, 0.95f, 0.85f);
    const float wristD = sdRoundCone(p, wrist, center + palmN * 1.6f, 2.7f, 2.9f);
    return smin(smin(smin(palm, fingers, 0.6f), thumb, 0.5f), wristD, 1.2f);
}

// Head in its own frame (x toward the golfer's left, y up, z out of the face),
// origin mid-cranium at ear height. Built from anthropometric landmarks for a
// 185 cm man (vertex to menton ~23.5 cm, head breadth ~15.5 cm) with Tiger's
// signature features: long face, strong square jaw and chin, high cheekbones,
// straight nasal bridge with a broad base, full lips, slightly prominent ears.
constant float3 kEyeL = float3( 3.15f, 1.0f, 7.75f);              // eyeball centres
constant float  kEyeRadius = 1.2f;

static float sdHead(float3 h)
{
    const float3 m = float3(abs(h.x), h.y, h.z);
    // Cranium and forehead.
    float d = sdEllipsoid(h - float3(0.0f, 2.6f, -0.7f), float3(7.8f, 9.0f, 9.9f));
    // Midface (maxilla) and the jaw: ramus, angle, body and a squared chin.
    d = smin(d, sdEllipsoid(h - float3(0.0f, -2.6f, 4.4f), float3(6.2f, 5.8f, 5.3f)), 2.5f);
    d = smin(d, sdEllipsoid(h - float3(0.0f, -6.6f, 3.6f), float3(5.5f, 4.4f, 5.0f)), 2.5f);
    d = smin(d, sdCapsule(m, float3(6.0f, -1.8f, -0.6f), float3(5.6f, -7.9f, 0.2f), 1.5f), 2.0f);     // ramus
    d = smin(d, sdCapsule(m, float3(5.6f, -7.9f, 0.2f), float3(2.3f, -10.4f, 6.4f), 1.6f), 2.2f);     // jaw line
    d = smin(d, sdEllipsoid(h - float3(0.0f, -9.6f, 7.8f), float3(3.4f, 2.3f, 2.3f)), 1.6f);          // chin
    // Cheekbones and brow.
    d = smin(d, sdEllipsoid(m - float3(4.7f, -0.5f, 6.6f), float3(2.4f, 1.4f, 2.2f)), 1.6f);
    d = smin(d, sdEllipsoid(h - float3(0.0f, 2.7f, 8.4f), float3(4.4f, 0.9f, 1.0f)), 2.0f);
    // Eye sockets, then the eyeballs and heavy upper lids (he is looking down).
    d = smax(d, -sdEllipsoid(m - float3(3.15f, 1.1f, 9.4f), float3(1.8f, 1.15f, 1.2f)), 0.8f);
    d = min(d, length(m - kEyeL) - kEyeRadius);
    d = smin(d, sdEllipsoid(m - float3(3.15f, 1.75f, 7.95f), float3(1.5f, 0.55f, 1.25f)), 0.5f);     // upper lids
    d = smin(d, sdEllipsoid(m - float3(3.15f, 0.15f, 8.0f), float3(1.45f, 0.32f, 1.1f)), 0.5f);      // lower lids
    // Nose: straight bridge to a rounded tip, broad alae, nostrils underneath.
    d = smin(d, sdRoundCone(h, float3(0.0f, 2.0f, 9.3f), float3(0.0f, -1.9f, 11.0f), 0.75f, 1.0f), 0.8f);
    d = smin(d, length(h - float3(0.0f, -2.2f, 10.9f)) - 1.15f, 0.6f);
    d = smin(d, sdEllipsoid(m - float3(1.5f, -2.55f, 9.8f), float3(1.05f, 0.85f, 0.95f)), 0.7f);
    d = smax(d, -sdEllipsoid(m - float3(0.8f, -3.1f, 10.4f), float3(0.38f, 0.22f, 0.45f)), 0.25f);
    // Mouth: philtrum, full upper and lower lips parted by a thin line.
    d = smin(d, sdEllipsoid(h - float3(0.0f, -4.0f, 9.5f), float3(1.8f, 1.0f, 0.9f)), 0.8f);
    d = smin(d, sdEllipsoid(h - float3(0.0f, -4.95f, 9.3f), float3(2.6f, 0.72f, 0.8f)), 0.8f);
    d = smin(d, sdEllipsoid(h - float3(0.0f, -6.15f, 9.1f), float3(2.4f, 0.72f, 0.85f)), 0.8f);
    d = smax(d, -sdEllipsoid(h - float3(0.0f, -5.55f, 9.85f), float3(2.4f, 0.09f, 0.8f)), 0.15f);
    // Ears: flat, tilted back a little, with a carved concha.
    const float3 e = m - float3(7.35f, -0.2f, -0.8f);
    const float3 er = float3(e.x, e.y * 0.97f + e.z * 0.24f, e.z * 0.97f - e.y * 0.24f);
    float ear = sdEllipsoid(er, float3(0.9f, 3.2f, 1.9f));
    ear = smax(ear, -sdEllipsoid(er - float3(0.85f, -0.3f, 0.2f), float3(0.4f, 1.5f, 0.9f)), 0.3f);
    return smin(d, ear, 0.6f);
}

// Golf shoe: white sole with toe spring, tapered leather upper, heel counter.
static float2 sdFoot(float3 p, float3 ankle, float3 dir)
{
    const float3 side = float3(dir.z, 0.0f, -dir.x);
    const float3 heel = float3(ankle.x, 0.0f, ankle.z) - dir * 6.0f;
    const float3 d    = p - (heel + dir * 15.0f);
    const float3 l    = float3(dot(d, side), d.y, dot(d, dir));
    const float  toe  = smoothstep(6.0f, 15.0f, l.z);                   // toe spring + taper
    const float  w    = 1.0f - 0.22f * toe;
    const float  sole  = sdRoundBox(float3(l.x / w, l.y - 1.0f - 0.9f * toe * toe, l.z), float3(5.6f, 1.0f, 15.3f), 0.9f) * w;
    const float  upper = sdRoundBox(float3(l.x / w, l.y - 4.8f + 0.12f * max(l.z, 0.0f) - 0.6f * toe, l.z - 0.3f),
                                    float3(5.0f, 3.4f, 14.6f), 3.0f) * w;
    const float  collar = sdRoundCone(p, ankle + float3(0, 5.0f, 0), heel + dir * 8.5f + float3(0, 6.5f, 0), 5.0f, 4.7f);
    return float2(min(sole, smin(upper, collar, 2.0f)), float(MAT_SHOE));
}

// World -> head frame (x toward the golfer's left, y up the head, z out of the face).
static float3 toHead(float3 p)
{
    const float3 h = p - kHead;
    return float3(dot(h, normalize(cross(kHeadUp, kHeadFace))), dot(h, kHeadUp), dot(h, kHeadFace));
}

static float2 mapGolfer(float3 p)
{
    const float3 t = toSpine(p);

    // Trousers: pelvis + glutes + both legs (cloth layer).
    const float3 tm = float3(abs(t.x), t.y, t.z);
    float pants = sdEllipsoid(t - float3(0.0f, 1.0f, -0.5f), float3(17.0f, 12.5f, 11.0f));
    pants = smin(pants, sdEllipsoid(tm - float3(7.5f, -2.5f, -3.8f), float3(8.0f, 8.8f, 5.8f)), 3.0f);   // seat
    pants = smin(pants, sdTrouserLeg(p, kHipLead,  kKneeLead,  kAnkleLead,   1.0f), 4.0f);
    pants = smin(pants, sdTrouserLeg(p, kHipTrail, kKneeTrail, kAnkleTrail, -1.0f), 4.0f);

    // Shirt: torso, shoulder girdle, sleeves and mock-neck collar.
    float shirt = sdShirtTorso(t);
    shirt = smin(shirt, sdCapsule(p, kShoulderLead, kShoulderTrail, 6.6f), 6.0f);
    shirt = smin(shirt, sdRoundCone(p, kNeck, kShoulderLead,  5.8f, 6.0f), 4.0f);   // trapezius
    shirt = smin(shirt, sdRoundCone(p, kNeck, kShoulderTrail, 5.8f, 6.0f), 4.0f);
    shirt = smin(shirt, sdDeltoid(p, kShoulderLead,  kElbowLead),  4.0f);
    shirt = smin(shirt, sdDeltoid(p, kShoulderTrail, kElbowTrail), 4.0f);
    shirt = smin(shirt, min(sdSleeve(p, kShoulderLead, kElbowLead), sdSleeve(p, kShoulderTrail, kElbowTrail)), 2.0f);
    const float collar = max(sdRoundCone(p, kNeck - kSpineUp * 4.0f, kNeck + kSpineUp * 2.5f, 8.2f, 7.2f),
                             -sdRoundCone(p, kNeck - kSpineUp * 6.0f, kNeck + kSpineUp * 6.0f, 6.1f, 5.9f));
    shirt = min(shirt, collar);

    // Tucked in at the belt: the belt line splits shirt / belt / trousers.
    const float clothD = smin(shirt, pants, 3.0f) * kClothLipschitz;
    const int   clothM = (t.y > 8.5f) ? MAT_SHIRT : ((t.y > 5.5f) ? MAT_BELT : MAT_PANTS);
    float2 res = float2(clothD, float(clothM));
    // Sleeves and collar stay shirt even where they dip below the belt plane.
    if (shirt * kClothLipschitz <= clothD + 1e-3f && t.y > 0.0f) res.y = float(MAT_SHIRT);

    // Skin: arms, neck, head (head only when near it).
    float skin = min(sdArm(p, kShoulderLead,  kElbowLead,  kWristLead,  kLeadHand),
                     sdArm(p, kShoulderTrail, kElbowTrail, kWristTrail, kTrailHand));
    const float3 hl = toHead(p);
    const float headBound = length(p - kHead) - 17.0f;
    const float neck = sdRoundCone(p, kNeck - kSpineUp * 4.0f, kHead - kHeadUp * 5.0f, 6.8f, 6.0f);
    skin = min(skin, (headBound < 2.0f) ? smin(neck, sdHead(hl), 2.0f) : min(neck, headBound));
    res = opU(res, float2(skin, float(MAT_SKIN)));

    // Hands on the grip (only evaluated near the grip).
    const float handBound = sdCapsule(p, kTrailHand - kShaftDir * 6.0f, kLeadHand + kShaftDir * 6.0f, 9.0f);
    if (handBound < 2.0f) {
        res = opU(res, float2(sdHand(p, kLeadHand,  kWristLead),  float(MAT_GLOVE)));   // gloved lead hand
        res = opU(res, float2(sdHand(p, kTrailHand, kWristTrail), float(MAT_SKIN)));
    } else {
        res = opU(res, float2(handBound, float(MAT_SKIN)));
    }

    // Cap: six-panel crown, button, pre-curved bill.
    if (headBound < 2.0f) {
        const float crown = max(sdEllipsoid(hl - float3(0.0f, 2.6f, -0.4f), float3(8.6f, 9.4f, 11.0f)),
                                -dot(hl - float3(0.0f, 2.1f, 0.0f), normalize(float3(0.0f, 1.0f, -0.08f))));
        const float button = length(hl - float3(0.0f, 11.9f, -0.8f)) - 0.6f;
        // D-shaped bill (~18 x 7.5 cm), pre-curved at the sides and pitched down.
        const float3 bl   = hl - float3(0.0f, 3.0f, 7.0f);
        const float  plan = max((length(float2(bl.x / 9.0f, bl.z / 7.5f)) - 1.0f) * 7.5f, -bl.z);
        const float  curl = (abs(bl.y + 0.035f * bl.x * bl.x + 0.10f * bl.z) - 0.18f) * 0.8f;
        const float  bill = length(max(float2(plan, curl), 0.0f)) + min(max(plan, curl), 0.0f);
        res = opU(res, float2(min(min(crown, button), bill), float(MAT_CAP)));
    }

    res = opU(res, sdFoot(p, kAnkleLead,  kFootDirLead));
    res = opU(res, sdFoot(p, kAnkleTrail, kFootDirTrail));
    return res;
}

// MARK: - Gallery ---------------------------------------------------------------

// Locates the nearest fan: returns their local position and cell id.
static float3 fanLocal(float3 p, int row, thread float2& id, thread float& height)
{
    const float rowZ = kFanRowZ0 - float(row) * kFanRowDZ;
    const float x    = p.x + float(row) * (kFanCellX * 0.5f);
    const float cell = round(x / kFanCellX);
    id = float2(cell, float(row));
    const float h1 = hash12(id * 1.37f + 0.5f);
    const float h2 = hash12(id * 2.11f + 4.1f);
    height = 160.0f + 30.0f * h1;
    const float3 base = float3(cell * kFanCellX - float(row) * (kFanCellX * 0.5f) + (h2 - 0.5f) * 20.0f,
                               terrainHeight(rowZ),
                               rowZ + (h1 - 0.5f) * 24.0f);
    return p - base;
}

static float2 sdFan(float3 l, float2 id, float H)
{
    if (hash12(id + 17.3f) < 0.12f) return float2(1e9f, 0.0f);   // gaps in the crowd
    const float3 lx = float3(abs(l.x), l.y, l.z);
    float2 res = float2(sdRoundCone(lx, float3(6.5f, 6, 0), float3(8.0f, 0.47f * H, 0), 5.5f, 8.0f), float(MAT_FAN_PANTS));
    const float3 lt = float3(l.x, l.y, l.z * 1.45f);
    float torso = sdRoundCone(lt, float3(0, 0.52f * H, 0), float3(0, 0.78f * H, 0), 15.0f, 18.0f) / 1.45f;
    torso = min(torso, sdCapsule(float3(abs(l.x), l.y, l.z), float3(19, 0.79f * H, 0), float3(21, 0.50f * H, 2), 5.0f));
    res = opU(res, float2(torso, float(MAT_FAN_SHIRT)));
    const float3 hc = float3(0, 0.91f * H, 0);
    res = opU(res, float2(length(l - hc) - 10.5f, float(MAT_FAN_SKIN)));
    if (hash12(id + 3.7f) < 0.55f) {
        const float hat = max(length(l - hc - float3(0, 1.5f, 0)) - 11.2f, -(l.y - hc.y - 2.0f));
        res = opU(res, float2(hat, float(MAT_FAN_HAT)));
    }
    return res;
}

static float2 mapGallery(float3 p)
{
    float2 res = float2(1e9f, 0.0f);

    // Gallery rope on stakes.
    const float sx   = p.x - 600.0f * round(p.x / 600.0f);
    const float rope = length(float2(p.y - 76.0f, p.z - kRopeZ)) - 1.0f;
    const float stake = sdCapsule(float3(sx, p.y, p.z - kRopeZ), float3(0, 0, 0), float3(0, 82, 0), 2.2f);
    res = opU(res, float2(min(rope, stake), float(MAT_ROPE)));

    // Rows of fans on the mound: test the two rows nearest to p.
    const float rowF = (kFanRowZ0 - p.z) / kFanRowDZ;
    const int r0 = clamp(int(floor(rowF)), 0, kFanRows - 1);
    const int r1 = clamp(r0 + 1, 0, kFanRows - 1);
    for (int k = 0; k < 2; ++k) {
        const int row = (k == 0) ? r0 : r1;
        float2 id; float H;
        const float3 l = fanLocal(p, row, id, H);
        res = opU(res, sdFan(l, id, H));
    }
    return res;
}

// MARK: - Augusta-style pines and azaleas --------------------------------------

static float2 mapTrees(float3 p)
{
    float2 res = float2(1e9f, 0.0f);

    // Azalea bank along the foot of the pines.
    {
        const float  cz   = kTreeZ0 + 140.0f;
        const float  cell = round(p.x / 260.0f);
        const float  h    = hash12(float2(cell, 7.0f));
        const float3 c    = float3(cell * 260.0f + (h - 0.5f) * 80.0f, terrainHeight(cz) + 40.0f, cz + (h - 0.5f) * 90.0f);
        const float  bumps = 12.0f * sin(p.x * 0.05f) * sin(p.y * 0.06f) * sin(p.z * 0.05f);
        res = opU(res, float2((sdEllipsoid(p - c, float3(170.0f, 105.0f, 130.0f)) + bumps) * 0.6f, float(MAT_AZALEA)));
    }

    // Understory: a dark wall of dogwood/holly between the azaleas and the pines.
    {
        const float  cz    = kTreeZ0 - 260.0f;
        const float  bumps = 40.0f * sin(p.x * 0.013f) * sin(p.y * 0.017f + 1.0f) * sin(p.z * 0.011f);
        const float  wall  = sdEllipsoid(float3(0.0f, p.y - terrainHeight(cz) - 250.0f, p.z - cz),
                                         float3(1.0f, 520.0f, 260.0f));
        res = opU(res, float2((wall + bumps) * 0.55f, float(MAT_CANOPY)));
    }

    // Tall pines: bare lower trunks, high canopies.
    const float rowF = (kTreeZ0 - p.z) / kTreeRowDZ;
    const int r0 = clamp(int(floor(rowF)), 0, kTreeRows - 1);
    const int r1 = clamp(r0 + 1, 0, kTreeRows - 1);
    for (int k = 0; k < 2; ++k) {
        const int   row  = (k == 0) ? r0 : r1;
        const float off  = float(row) * 0.37f * kTreeCellX;
        const float cell = round((p.x + off) / kTreeCellX);
        const float2 id  = float2(cell, float(row));
        const float h1   = hash12(id * 1.9f + 2.0f), h2 = hash12(id * 3.3f + 9.0f);
        const float z    = kTreeZ0 - float(row) * kTreeRowDZ + (h2 - 0.5f) * 260.0f;
        const float3 base = float3(cell * kTreeCellX - off + (h1 - 0.5f) * 260.0f, terrainHeight(z), z);
        const float  height = 2000.0f + 700.0f * h1;
        const float3 l = p - base;
        res = opU(res, float2(sdCappedCone(l, float3(0, -50, 0), float3(0, height, 0), 34.0f, 12.0f), float(MAT_TRUNK)));

        const float bumps = 30.0f * sin(l.x * 0.021f + h1 * 6.0f) * sin(l.y * 0.027f) * sin(l.z * 0.023f + h2 * 5.0f);
        float canopy = sdEllipsoid(l - float3(40.0f * h2, height * 0.52f, 0.0f), float3(360.0f, 300.0f, 340.0f));
        canopy = smin(canopy, sdEllipsoid(l - float3(-60.0f * h1, height * 0.76f, 30.0f), float3(330.0f, 260.0f, 310.0f)), 80.0f);
        canopy = smin(canopy, sdEllipsoid(l - float3(20.0f, height * 0.97f, -20.0f), float3(230.0f, 170.0f, 230.0f)), 80.0f);
        res = opU(res, float2((canopy + bumps) * 0.6f, float(MAT_CANOPY)));
    }
    return res;
}

// MARK: - Staff bag -------------------------------------------------------------

// Bag frame: x = side, y = along the bag, z = toward the front panel (+Z).
static float3 toBag(float3 p)
{
    const float3 fwd  = normalize(float3(0, 0, 1) - kBagUp * kBagUp.z);
    const float3 side = cross(kBagUp, fwd);
    const float3 d    = p - kBagBase;
    return float3(dot(d, side), dot(d, kBagUp), dot(d, fwd));
}

static float2 mapBag(float3 p)
{
    const float3 b = toBag(p);
    // Body, top cuff and front ball pocket.
    float body = sdCappedCone(b, float3(0, 1.5f, 0), float3(0, kBagHeight, 0), kBagRadius - 1.0f, kBagRadius);
    body = smin(body, sdRoundBox(b - float3(0, 38.0f, kBagRadius - 1.0f), float3(9.0f, 20.0f, 3.0f), 3.0f), 2.0f);
    const float cuff = sdCappedCone(b, float3(0, kBagHeight - 6.0f, 0), float3(0, kBagHeight + 1.5f, 0),
                                    kBagRadius + 1.2f, kBagRadius + 1.6f);
    float2 res = float2(min(body, cuff), float(MAT_BAG));

    // Headcovers for driver, 3-wood and hybrid at the back of the bag.
    float covers = 1e9f;
    for (int i = 0; i < 3; ++i) {
        const float  x = -8.0f + 8.0f * float(i);
        const float3 c = float3(x, kBagHeight + 14.0f + 3.0f * float(i == 1), -6.0f);
        covers = min(covers, sdEllipsoid(b - c, float3(6.2f, 9.5f, 7.0f)));
        covers = min(covers, sdCapsule(b, float3(x, kBagHeight, -6.0f), c, 3.2f));
    }
    res = opU(res, float2(covers, float(MAT_HEADCOVER)));

    // Irons (4-PW) fanned across the front: short shaft stubs plus heads.
    float irons = 1e9f;
    for (int i = 0; i < 7; ++i) {
        const float  x    = -10.5f + 3.5f * float(i);
        const float  rise = 6.0f + 0.9f * float(6 - i);                 // longer irons sit higher
        const float3 top  = float3(x, kBagHeight + rise, 6.0f);
        irons = min(irons, sdCapsule(b, float3(x, kBagHeight - 2.0f, 5.0f), top, 0.45f));
        irons = min(irons, sdRoundBox(b - top - float3(0.0f, 1.8f, 2.4f), float3(0.9f, 2.2f, 3.6f), 0.5f));
    }
    res = opU(res, float2(irons, float(MAT_BAG_IRON)));

    // Tour towel hung over the cuff beside the script, folded over the rim.
    const float r     = length(b.xz);
    const float ang   = atan2(b.x, b.z);
    const float along = (ang - 0.95f) * r;                                     // across the towel (cm)
    const float folds = 0.45f * sin(along * 0.85f + 0.6f) * smoothstep(kBagHeight - 2.0f, kBagHeight - 20.0f, b.y);
    const float hem   = 1.2f * sin(along * 0.7f + 1.3f);
    const float drape = max(max(abs(r - (kBagRadius + 1.7f) - folds) - 0.30f, abs(along) - 4.6f),
                            abs(b.y - (kBagHeight - 13.0f) + 0.5f * hem) - 15.0f) - 0.25f;
    const float over  = max(max(abs(b.y - (kBagHeight + 2.0f)) - 0.45f, (abs(ang - 0.95f) - 0.30f) * r),
                            abs(r - (kBagRadius + 0.9f)) - 1.8f);
    res = opU(res, float2(min(drape, over) * 0.8f, float(MAT_TOWEL)));
    return res;
}

// MARK: - Scene -----------------------------------------------------------------

static float2 mapScene(float3 p)
{
    // Terrain: a height field in z only. Scaling by 0.9 keeps it a safe bound on the slopes.
    float2 res = float2((p.y - terrainHeight(p.z)) * 0.9f, float(MAT_TERRAIN));

    // Each group is evaluated only when p is near its bounding volume. The
    // switch distance exceeds the AO probe reach (ambientOcclusion), so AO
    // never samples a bound instead of the real surface.
    const float bHero = length(p - float3(0.0f, 3.0f, 0.0f)) - 10.0f;
    res = opU(res, (bHero < 14.0f) ? mapHero(p) : float2(bHero, float(MAT_HEAD)));

    const float bShaft = sdCapsule(p, hoselTop(), kGripEnd, 3.0f);
    res = opU(res, (bShaft < 14.0f) ? mapShaft(p) : float2(bShaft, float(MAT_SHAFT)));

    const float bGolfer = sdBox(p - float3(-5.0f, 85.0f, -73.0f), float3(46.0f, 86.0f, 46.0f));
    res = opU(res, (bGolfer < 14.0f) ? mapGolfer(p) : float2(bGolfer, float(MAT_SHIRT)));

    const float bBag = length(p - (kBagBase + kBagUp * 60.0f)) - 75.0f;
    res = opU(res, (bBag < 14.0f) ? mapBag(p) : float2(bBag, float(MAT_BAG)));

    // Augusta-green tee markers and the leaderboard on its posts: cheap boxes, always evaluated.
    const float markers = min(sdRoundBox(p - kTeeMarkerA - float3(0, 5.0f, 0), float3(7.0f, 4.0f, 11.0f), 2.0f),
                              sdRoundBox(p - kTeeMarkerB - float3(0, 5.0f, 0), float3(7.0f, 4.0f, 11.0f), 2.0f));
    res = opU(res, float2(markers, float(MAT_MARKER)));
    const float3 bc = 0.5f * (kBoardMin + kBoardMax);
    float board = sdBox(p - bc, 0.5f * (kBoardMax - kBoardMin));
    for (int i = 0; i < 2; ++i) {
        const float px = mix(kBoardMin.x + 90.0f, kBoardMax.x - 90.0f, float(i));
        board = min(board, sdBox(p - float3(px, 0.5f * kBoardMin.y + 60.0f, kBoardMin.z - 10.0f),
                                 float3(12.0f, 0.5f * kBoardMin.y - 60.0f, 12.0f)));
    }
    res = opU(res, float2(board, float(MAT_BOARD)));

    const float bGallery = sdBox(p - float3(0.0f, 180.0f, -1350.0f), float3(1e5f, 220.0f, 330.0f));
    res = opU(res, (bGallery < 64.0f) ? mapGallery(p) : float2(bGallery, float(MAT_ROPE)));

    const float bTrees = sdBox(p - float3(0.0f, 1800.0f, -3400.0f), float3(1e5f, 1900.0f, 1400.0f));
    res = opU(res, (bTrees < 64.0f) ? mapTrees(p) : float2(bTrees, float(MAT_TRUNK)));
    return res;
}

static float3 calcNormal(float3 p, float eps)
{
    const float2 k = float2(1.0f, -1.0f);
    return normalize(k.xyy * mapScene(p + k.xyy * eps).x +
                     k.yyx * mapScene(p + k.yyx * eps).x +
                     k.yxy * mapScene(p + k.yxy * eps).x +
                     k.xxx * mapScene(p + k.xxx * eps).x);
}

// Classic SDF soft shadow. The start distance is jittered per sample, which
// turns step banding into fine noise that the pixel samples average out.
// (The "improved" two-sphere estimator assumes one continuous field and
// misfires at this scene's bounding-volume switches.)
static float softShadow(float3 ro, float3 rd, float k, float tStart)
{
    float res = 1.0f;
    float t   = tStart * (1.0f + hash12(ro.xz * 13.7f + ro.y));
    for (int i = 0; i < 128 && t < 3000.0f; ++i) {
        const float h = mapScene(ro + rd * t).x;
        res = min(res, k * h / t);
        if (res < 0.002f) break;
        t += clamp(h, 0.01f + 0.004f * t, 60.0f);
    }
    return smoothstep(0.0f, 1.0f, saturate(res));
}

// AO sampled along the normal; `scale` matches the size of the local features.
static float ambientOcclusion(float3 p, float3 n, float scale)
{
    float occ = 0.0f, weight = 1.0f;
    for (int i = 0; i < 5; ++i) {
        const float h = scale * (0.04f + float(i) / 4.0f);
        occ    += (h - mapScene(p + n * h).x) / scale * weight;
        weight *= 0.72f;
    }
    return saturate(1.0f - 0.9f * occ);
}

// MARK: - Environment -----------------------------------------------------------

static float3 environment(float3 rd)
{
    const float3 L = normalize(kSunDir);
    if (rd.y >= 0.0f) {
        const float mu = saturate(dot(rd, L));
        float3 sky = mix(kSkyHorizon, kSkyTop, pow(saturate(rd.y), 0.5f));
        sky += kSunColor * (0.06f * pow(mu, 6.0f) + 0.25f * pow(mu, 64.0f));   // forward scattering
        return sky;
    }
    return mix(kSkyHorizon * 0.7f, kGroundBounce, smoothstep(0.0f, 0.2f, -rd.y));
}

// Cosine-weighted sky + turf bounce seen by a surface with normal n.
static float3 irradiance(float3 n)
{
    const float3 sky = mix(kSkyHorizon * 0.85f, kSkyTop, saturate(n.y));
    return mix(kGroundBounce, sky, n.y * 0.5f + 0.5f) * 0.6f;
}

// MARK: - Dimples ---------------------------------------------------------------

static float4 nearestFibonacciPoint(float3 p, float n)
{
    const float m        = 1.0f - 1.0f / n;
    const float phi      = min(atan2(p.y, p.x), M_PI_F);
    const float cosTheta = p.z;
    const float sin2     = max(1e-6f, 1.0f - cosTheta * cosTheta);
    const float k        = max(2.0f, floor(log(n * M_PI_F * sqrt(5.0f) * sin2) / log(kPHI * kPHI)));
    const float Fk       = pow(kPHI, k) / sqrt(5.0f);
    const float2 F       = float2(round(Fk), round(Fk * kPHI));
    const float2 ka      = 2.0f * F / n;
    const float2 kb      = 2.0f * M_PI_F * (fract((F + 1.0f) * kPHI) - (kPHI - 1.0f));
    const float2x2 iB    = float2x2(float2(ka.y, -ka.x),
                                    float2(kb.y, -kb.x)) / (ka.y * kb.x - ka.x * kb.y);
    const float2 c       = floor(iB * float2(phi, cosTheta - m));

    float  bestD2 = 8.0f;
    float3 bestQ  = float3(0.0f, 0.0f, 1.0f);
    for (int s = 0; s < 4; ++s) {
        const float2 uv   = float2(float(s & 1), float(s >> 1));
        const float  i    = clamp(round(dot(F, uv + c)), 0.0f, n - 1.0f);
        const float  qPhi = 2.0f * M_PI_F * fract(i * kPHI);
        const float  qCos = m - 2.0f * i / n;
        const float  qSin = sqrt(max(0.0f, 1.0f - qCos * qCos));
        const float3 q    = float3(cos(qPhi) * qSin, sin(qPhi) * qSin, qCos);
        const float  d2   = distance_squared(q, p);
        if (d2 < bestD2) { bestD2 = d2; bestQ = q; }
    }

    // Near the poles the lattice basis degenerates; check the polar points directly.
    if (abs(cosTheta) > 0.93f) {
        for (int j = 0; j < 16; ++j) {
            const float  i    = (cosTheta > 0.0f) ? float(j) : n - 1.0f - float(j);
            const float  qPhi = 2.0f * M_PI_F * fract(i * kPHI);
            const float  qCos = m - 2.0f * i / n;
            const float  qSin = sqrt(max(0.0f, 1.0f - qCos * qCos));
            const float3 q    = float3(cos(qPhi) * qSin, sin(qPhi) * qSin, qCos);
            const float  d2   = distance_squared(q, p);
            if (d2 < bestD2) { bestD2 = d2; bestQ = q; }
        }
    }
    return float4(bestQ, sqrt(bestD2));
}

static float4 dimpleNormal(float3 nWorld, float dimpleCount, float footprint)
{
    // Lattice poles point along (-1,0,1)/(1,0,-1): edge-on or hidden for every preset.
    const float3x3 toWorld  = rotationAxisAngle(float3(0.0f, 1.0f, 0.0f), -0.785398f);
    const float3x3 toObject = transpose(toWorld);

    const float3 nObj    = toObject * nWorld;
    const float4 dm      = nearestFibonacciPoint(nObj, dimpleCount);
    const float  spacing = sqrt(8.0f * M_PI_F / (sqrt(3.0f) * dimpleCount));
    const float  rDimple = kDimpleRadiusScale * spacing;
    const float  lod     = 1.0f - smoothstep(0.25f * rDimple, 0.75f * rDimple, footprint);

    const float t = dm.w / rDimple;
    if (t >= 1.0f || lod <= 0.0f) return float4(nWorld, 1.0f);

    float3 o = nObj - dm.xyz;
    o -= nObj * dot(o, nObj);
    const float tilt   = kDimpleSlope * (1.0f - smoothstep(0.88f, 1.0f, t)) * lod / rDimple;
    const float cavity = mix(1.0f, mix(kDimpleCavityAO, 1.0f, t * t), lod);
    return float4(toWorld * normalize(nObj - o * tilt), cavity);
}

// MARK: - Physically based shading ---------------------------------------------
// One model for every material: GGX microfacet specular with Schlick Fresnel,
// Lambert diffuse, roughness-aware sky reflections (split-sum approximation),
// plus optional sheen (cloth) and wrap/subsurface (skin, leaves, grass).

struct Surface {
    float3 albedo;       // base colour; for metals, the specular colour
    float  metallic;
    float  roughness;    // perceptual roughness, 0 = mirror
    float  sheen;        // cloth fuzz at grazing angles
    float  subsurface;   // light wrapping past the terminator, tinted warm
};

static Surface dielectric(float3 albedo, float roughness, float sheen = 0.0f, float subsurface = 0.0f)
{
    return Surface{ albedo, 0.0f, roughness, sheen, subsurface };
}

static Surface metallic(float3 f0, float roughness) { return Surface{ f0, 1.0f, roughness, 0.0f, 0.0f }; }

struct Light {
    float3 L;
    float  shadow;
    float  ao;
};

// Pre-integrated GGX environment BRDF (Karis 2014, analytic fit).
static float3 envBRDF(float3 f0, float roughness, float NoV)
{
    const float4 c0 = float4(-1.0f, -0.0275f, -0.572f, 0.022f);
    const float4 c1 = float4( 1.0f,  0.0425f,  1.040f, -0.040f);
    const float4 r  = roughness * c0 + c1;
    const float  a  = min(r.x * r.x, exp2(-9.28f * NoV)) * r.x + r.y;
    const float2 ab = float2(-1.04f, 1.04f) * a + r.zw;
    return f0 * ab.x + ab.y;
}

static float3 shade(Surface s, float3 n, float3 rd, Light li)
{
    const float3 V    = -rd;
    const float3 H    = normalize(li.L + V);
    const float  NoV  = max(dot(n, V), 1e-4f);
    const float  NoLr = dot(n, li.L);
    const float  NoL  = saturate(NoLr);
    const float  NoH  = saturate(dot(n, H));
    const float  VoH  = saturate(dot(V, H));
    const float  a    = max(s.roughness * s.roughness, 0.002f);
    const float3 f0   = mix(float3(0.04f), s.albedo, s.metallic);
    const float3 kd   = s.albedo * (1.0f - s.metallic);
    const float3 sun  = kSunColor * li.shadow;

    // Direct sun: GGX distribution, Hammon's height-correlated Smith
    // visibility, Schlick Fresnel. The factor pi matches kSunColor's convention.
    const float  dd  = NoH * NoH * (a * a - 1.0f) + 1.0f;
    const float  D   = a * a / (M_PI_F * dd * dd);
    const float  Vis = 0.5f / max(mix(2.0f * NoL * NoV, NoL + NoV, a), 1e-4f);
    const float3 F   = f0 + (1.0f - f0) * pow(1.0f - VoH, 5.0f);
    float3 col = D * Vis * F * (M_PI_F * NoL) * sun;

    // Diffuse; subsurface materials wrap light past the terminator, tinted warm.
    const float w      = 0.5f * s.subsurface;
    const float wrap   = saturate((NoLr + w) / ((1.0f + w) * (1.0f + w)));
    const float3 tint  = mix(float3(1.0f), kScatterTint, s.subsurface * saturate(1.0f - NoLr * 2.0f));
    const float3 Fa    = envBRDF(f0, s.roughness, NoV);
    col += kd * (1.0f - Fa) * (sun * wrap * tint + irradiance(n) * li.ao);

    // Sky reflection: rough surfaces see a blurrier, irradiance-like sky.
    // Specular occlusion follows Lagarde & de Rousiers 2014.
    const float3 R      = reflect(rd, n);
    const float3 prefil = mix(environment(R), irradiance(R) * 1.6f, sqrt(s.roughness));
    const float  specAO = saturate(pow(NoV + li.ao, exp2(-16.0f * a - 1.0f)) - 1.0f + li.ao);
    col += prefil * Fa * specAO;

    // Cloth sheen: soft retro-reflective rim.
    col += s.sheen * pow(1.0f - NoV, 4.0f) * mix(float3(1.0f), s.albedo, 0.5f)
         * (sun * wrap * 0.25f + irradiance(n) * li.ao * 0.6f);
    return col;
}

// MARK: - Material: T100-style head --------------------------------------------

static Surface clubHeadSurface(float3 pos, float3 n)
{
    const float3 q = pos - float3(kLeadEdgeX, kSoleLift, 0.0f);
    const float  u = q.z;
    const float  s = q.x + q.y * kLoftTan;
    const float3 faceNormal = float3(kLoftCos, kLoftTan * kLoftCos, 0.0f);

    // Face: bead-blasted steel; score lines are darker, rougher channels.
    if (s > -0.03f && dot(n, faceNormal) > 0.9f) {
        const float up     = q.y / kLoftCos;
        const bool  inArea = u > kHeelZ + 1.1f && u < kToeZ - 1.1f && up > 0.45f;
        const float gv     = abs(fract(up / 0.36f) - 0.5f) * 0.36f;
        const float groove = inArea ? 1.0f - smoothstep(0.022f, 0.040f, gv) : 0.0f;
        return metallic(float3(0.56f, 0.56f, 0.57f) * (1.0f - 0.6f * groove), mix(0.42f, 0.7f, groove));
    }

    if (n.x < -0.35f) {
        // Badge insert in the cavity: gunmetal with white "T100" and a red keyline.
        const float2 bp    = float2(u, q.y) - kBadgeCenter;
        const float  badge = sdRoundBox2(bp, kBadgeHalf, float4(0.35f));
        if (badge < 0.0f) {
            const float capH = 0.52f;
            const float w    = textWidth(kTextT100, 4, 0.12f) * capH;
            const float txt  = sdText((bp - float2(-0.5f * w, -0.5f * capH)) / capH, kTextT100, 4, 0.12f, 0.0f) * capH;
            if (txt < 0.045f)    return dielectric(float3(0.85f), 0.4f);              // paint fill
            if (badge > -0.06f)  return dielectric(float3(0.62f, 0.03f, 0.03f), 0.4f);
            return metallic(float3(0.20f, 0.20f, 0.21f), 0.3f);
        }

        // Lower muscle bar: script wordmark and the club number in black paint.
        const float capS   = 0.62f;
        const float script = sdText((float2(u, q.y) - float2(kHeelZ + 1.05f, 0.62f)) / capS,
                                    kTextTitleist, 8, 0.02f, 0.28f) * capS;
        const float four   = sdText((float2(u, q.y) - float2(kToeZ - 1.15f, 0.62f)) / capS,
                                    kTextFour, 1, 0.0f, 0.12f) * capS;
        if (min(script, four) < 0.045f) return dielectric(float3(0.02f), 0.4f);

        // Satin cavity, undercut and channel inside a mirror-polished perimeter.
        const bool satin = (q.y > muscleLine(u) - 0.6f && bladeProfile(u, q.y) < -kFrameWidth)
                        || (q.y < 0.56f && s < -1.7f);
        return metallic(float3(0.62f, 0.63f, 0.65f), satin ? 0.35f : 0.10f);
    }
    return metallic(float3(0.64f, 0.65f, 0.67f), 0.10f);                  // polished chrome
}

// MARK: - Grass blades (close-up geometry) --------------------------------------
// Near the camera the tee box is real geometry: one mowed perennial-rye blade
// per 4.2 mm cell (~6 shoots/cm²), ~1.3 cm tall, a flat two-segment ribbon
// leaning with its mowing stripe, with a blunt cut tip. Blades live in a thin
// slab above y = 0 and are traced in their own pass (traceGrass), so the main
// scene SDF stays cheap and distant views never pay for them.

constant float kGrassCell   = 0.42f;
constant float kGrassHeight = 1.3f;

// Direction the blades lean on the tee box's diagonal mowing stripes.
static float2 mowLean(float2 xz)
{
    const bool  teeBox = abs(xz.x) < 650.0f && xz.y > -600.0f && xz.y < 650.0f;
    const float side   = sin((xz.x + xz.y) * (M_PI_F / 150.0f));
    return (teeBox ? sign(side) : 1.0f) * float2(0.7071f, 0.7071f);
}

// Blade geometry fades out 90-160 cm from the camera, where blades go sub-pixel.
static float grassFade(float2 xz, float3 eye)
{
    return 1.0f - smoothstep(90.0f, 160.0f, distance(xz, eye.xz));
}

static float sdGrassBlade(float3 p, float2 cell, float2 lean, float fade, thread float& hFrac)
{
    const float h1 = hash12(cell), h2 = hash12(cell + 17.1f), h3 = hash12(cell + 41.7f), h4 = hash12(cell + 73.3f);
    const float2 base  = (cell + 0.25f + 0.5f * float2(h1, h2)) * kGrassCell;
    const float  H     = kGrassHeight * (0.55f + 0.7f * h3) * fade;
    const float2 dir   = normalize(lean + 0.75f * float2(cos(h4 * 6.2832f), sin(h4 * 6.2832f)));
    const float  reach = (0.20f + 0.30f * h2) * H;   // stays within the 3x3 cell search
    const float3 b0 = float3(base.x, -0.05f, base.y);
    const float3 b1 = b0 + float3(dir.x * reach * 0.3f, 0.6f * H, dir.y * reach * 0.3f);
    const float3 b2 = b0 + float3(dir.x * reach, H, dir.y * reach);

    // Closest point on the two-segment spine.
    const float3 pa0 = p - b0, ba0 = b1 - b0;
    const float3 pa1 = p - b1, ba1 = b2 - b1;
    const float  s0  = saturate(dot(pa0, ba0) / dot(ba0, ba0));
    const float  s1  = saturate(dot(pa1, ba1) / dot(ba1, ba1));
    const float3 v0  = pa0 - ba0 * s0, v1 = pa1 - ba1 * s1;
    const bool   top = dot(v1, v1) < dot(v0, v0);
    const float3 v   = top ? v1 : v0;
    hFrac = top ? 0.6f + 0.4f * s1 : 0.6f * s0;

    // Flat ribbon: tapering half-width across the lean, thin through it.
    const float3 w      = float3(-dir.y, 0.0f, dir.x);
    const float  across = dot(v, w);
    const float  hw     = 0.085f * (1.0f - 0.3f * hFrac);
    return length(float2(max(abs(across) - hw, 0.0f), length(v - w * across))) - 0.012f;
}

static float sdGrass(float3 p, float fade, thread float& hFrac, thread float2& bladeId)
{
    if (fade < 0.02f) return p.y + 1.0f;
    const float2 c0   = floor(p.xz / kGrassCell);
    const float2 lean = mowLean(p.xz);
    float d = 1e9f;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            float hf;
            const float2 cell = c0 + float2(i, j);
            const float  db   = sdGrassBlade(p, cell, lean, fade, hf);
            if (db < d) { d = db; hFrac = hf; bladeId = cell; }
        }
    }
    return d;
}

// Marches the blade slab between its top and min(ground, tLimit). Returns t or -1.
static float traceGrass(float3 ro, float3 rd, float3 eye, float tLimit,
                        thread float& hFrac, thread float2& bladeId)
{
    if (rd.y >= 0.0f) return -1.0f;
    float       t    = max((kGrassHeight * 1.25f - ro.y) / rd.y, 0.0f);
    const float tEnd = min(-ro.y / rd.y, tLimit);
    if (t >= tEnd || max(grassFade((ro + rd * t).xz, eye), grassFade((ro + rd * tEnd).xz, eye)) <= 0.0f) {
        return -1.0f;
    }
    for (int i = 0; i < 160 && t < tEnd; ++i) {
        const float3 p = ro + rd * t;
        const float  d = sdGrass(p, grassFade(p.xz, eye), hFrac, bladeId);
        if (d < 0.0015f) return t;
        t += max(d * 0.9f, 0.003f);
    }
    return -1.0f;
}

static float3 shadeGrassBlade(float3 pos, float3 rd, float3 eye, float hFrac, float2 bladeId)
{
    // Normal from the blade field; ribbons are two-sided, so face the viewer.
    const float  fade = grassFade(pos.xz, eye);
    const float2 k = float2(1.0f, -1.0f);
    const float  e = 0.002f;
    float hf; float2 id;
    float3 n = normalize(k.xyy * sdGrass(pos + k.xyy * e, fade, hf, id) + k.yyx * sdGrass(pos + k.yyx * e, fade, hf, id) +
                         k.yxy * sdGrass(pos + k.yxy * e, fade, hf, id) + k.xxx * sdGrass(pos + k.xxx * e, fade, hf, id));
    if (dot(n, rd) > 0.0f) n = -n;

    float3 albedo = mix(float3(0.045f, 0.16f, 0.025f), float3(0.11f, 0.30f, 0.05f), hash12(bladeId + 5.0f));
    albedo = mix(albedo * 0.55f, albedo, hFrac);                                         // older growth low down
    albedo = mix(albedo, float3(0.50f, 0.50f, 0.36f), 0.35f * smoothstep(0.95f, 1.0f, hFrac)); // frayed mower-cut tip

    // Neighbouring blades shade and occlude the lower part of each blade.
    Light li;
    li.L      = normalize(kSunDir);
    li.shadow = softShadow(pos + n * 0.01f, li.L, 12.0f, 0.03f) * mix(0.5f, 1.0f, hFrac);
    li.ao     = mix(0.4f, 1.0f, pow(hFrac, 0.7f));
    return shade(dielectric(albedo, 0.5f, 0.1f, 0.8f), n, rd, li);
}

// MARK: - Material: turf --------------------------------------------------------

// Tee-box turf. Mowing stripes are blades leaning in alternating directions,
// so they lighten when mowed away from the viewer and darken when mowed toward.
static Surface turf(float3 pos, float t, float3 rd, thread float3& n)
{
    const float2 xz   = pos.xz;
    const float  fine = saturate(1.0f - t / 600.0f);              // fade blade detail with distance

    const bool  teeBox  = abs(pos.x) < 650.0f && pos.z > -600.0f && pos.z < 650.0f;
    const float side    = sin((pos.x + pos.z) * (M_PI_F / 150.0f));
    const float2 lean   = mowLean(xz);

    // Blade micro-relief: noise stretched along the lean direction.
    const float2 along = float2(dot(xz, lean), dot(xz, float2(-lean.y, lean.x)));
    const float  e  = 0.04f;
    const float  h0 = fbm(along * float2(1.5f, 7.0f));
    const float  hx = fbm((along + float2(e, 0)) * float2(1.5f, 7.0f));
    const float  hz = fbm((along + float2(0, e)) * float2(1.5f, 7.0f));
    const float2 g  = float2(hx - h0, hz - h0) / e * 0.10f * fine;
    const float2 gw = lean * g.x + float2(-lean.y, lean.x) * g.y;
    n = normalize(float3(-gw.x + 0.25f * lean.x, 1.0f, -gw.y + 0.25f * lean.y));

    // Overseeded rye: deep green with lighter tips and slight patchiness.
    const float patches = fbm(xz * 0.01f);
    const float blade   = hash12(floor(along * float2(3.0f, 14.0f)));
    float3 c = mix(float3(0.045f, 0.17f, 0.025f), float3(0.10f, 0.31f, 0.045f), patches);
    c *= mix(1.0f, 0.80f + 0.40f * blade, fine);
    c = mix(c, c * float3(1.15f, 1.10f, 0.80f), 0.35f * h0 * fine);   // sun-bleached tips

    if (teeBox) {
        // Stripe brightness from blade lean relative to the view direction.
        const float2 view = normalize(rd.xz + 1e-5f);
        c *= 1.0f + 0.16f * dot(view, lean) * smoothstep(0.0f, 0.15f, abs(side));

        // Sunday tee box: divot scrapes filled with Augusta's green-tinted
        // sand-and-seed mix, clustered in the hitting zone and running along the
        // target line; none at the player's fresh spot around the tee.
        const float2 cell  = floor(xz / 32.0f);
        const float  hd    = hash12(cell + 11.0f);
        const float2 cpos  = (cell + 0.25f + 0.5f * float2(hd, hash12(cell + 23.0f))) * 32.0f;
        const float  zone  = (1.0f - smoothstep(150.0f, 260.0f, abs(cpos.x + 40.0f)))
                           * (1.0f - smoothstep(120.0f, 220.0f, abs(cpos.y + 40.0f)));
        const float2 dq    = (xz - cpos) / float2(7.0f, 2.6f);
        const float  ragged = 0.15f * valueNoise(xz * 0.9f);
        const float  divot = (hd < 0.18f * zone && length(cpos) > 70.0f)
                           ? 1.0f - smoothstep(0.7f, 1.0f, length(dq) + ragged) : 0.0f;
        const float3 sand  = mix(float3(0.20f, 0.26f, 0.11f), float3(0.08f, 0.22f, 0.05f),
                                 step(0.7f, hash12(floor(xz * 6.0f))));         // germinating seed
        c = mix(c, sand, 0.85f * divot);
    } else {
        c *= 0.85f;                                                        // longer second cut
    }

    // Scuffed soil where the tee went in.
    const float r = length(xz);
    c = mix(float3(0.24f, 0.18f, 0.11f), c, smoothstep(0.25f, 0.8f, r + 0.3f * blade));

    // Pine straw under the trees.
    const float straw = smoothstep(kTreeZ0 + 250.0f, kTreeZ0 - 50.0f, pos.z);
    c = mix(c, float3(0.36f, 0.20f, 0.08f) * (0.75f + 0.5f * patches), straw);

    // Grass is translucent and has a soft waxy sheen.
    // A lawn is a canopy of blades, not a glossy plane: high roughness keeps
    // grazing-angle Fresnel from greying it out.
    return dielectric(c, mix(0.8f, 0.9f, straw), 0.12f * (1.0f - straw), 0.6f * (1.0f - straw));
}

// MARK: - Gallery colours -------------------------------------------------------

static float3 fanColor(float3 pos, int mat)
{
    // Recover which fan this is (nearest of the two candidate rows).
    const float rowF = (kFanRowZ0 - pos.z) / kFanRowDZ;
    const int   row  = clamp(int(round(rowF)), 0, kFanRows - 1);
    float2 id; float H;
    fanLocal(pos, row, id, H);
    const float h = hash12(id * 5.3f + 1.0f);
    const float g = hash12(id * 7.1f + 2.0f);

    switch (mat) {
        case MAT_FAN_SHIRT: {
            const float3 palette[8] = { float3(0.85f, 0.85f, 0.82f), float3(0.45f, 0.62f, 0.85f),
                                        float3(0.90f, 0.55f, 0.65f), float3(0.90f, 0.82f, 0.40f),
                                        float3(0.20f, 0.42f, 0.25f), float3(0.12f, 0.16f, 0.32f),
                                        float3(0.55f, 0.78f, 0.50f), float3(0.88f, 0.50f, 0.38f) };
            return palette[int(h * 7.999f)] * 0.8f;
        }
        case MAT_FAN_PANTS: return (g < 0.5f) ? float3(0.55f, 0.47f, 0.33f) : float3(0.14f, 0.17f, 0.26f);
        case MAT_FAN_SKIN:  return mix(float3(0.62f, 0.45f, 0.34f), float3(0.25f, 0.15f, 0.10f), g * g);
        default:            return (h < 0.5f) ? float3(0.86f) : float3(0.12f, 0.33f, 0.18f);  // caps
    }
}

// MARK: - Per-sample render -----------------------------------------------------

static float3 renderSample(float2 pixel, float2 resolution, float2 lensSample,
                           constant GolfSceneUniforms& u, float subpixelScale)
{
    float2 uv = (2.0f * pixel - resolution) / resolution.y;
    uv.y = -uv.y;                                          // textures are y-down

    const float3 eye   = u.cameraPosition.xyz;
    const float  focal = u.cameraTarget.w;
    const float3 fw    = normalize(u.cameraTarget.xyz - eye);
    const float3 rt    = normalize(cross(fw, float3(0.0f, 1.0f, 0.0f)));
    const float3 up    = cross(rt, fw);
    const float3 rdPin = normalize(uv.x * rt + uv.y * up + focal * fw);

    // Thin lens: start on the aperture disk, aim at the focal plane.
    const float3 focusPoint = eye + rdPin * (u.focusDistance / dot(rdPin, fw));
    const float3 ro = eye + (rt * lensSample.x + up * lensSample.y) * u.cameraPosition.w;
    const float3 rd = normalize(focusPoint - ro);

    // The terrain is flat (y = 0) in front of the rope, so a ray that reaches
    // y = 0 there hits it exactly. That bounds the march, and grazing rays
    // never run out of steps.
    const float tPlane   = (rd.y < 0.0f) ? -ro.y / rd.y : INFINITY;
    const bool  flatHit  = rd.y < 0.0f && (ro.z + rd.z * tPlane) > -1120.0f;
    const float tMax     = flatHit ? tPlane : kMaxDistance;

    float t   = 0.0f;
    int   mat = MAT_NONE;
    for (int i = 0; i < kMaxSteps; ++i) {
        const float2 h = mapScene(ro + rd * t);
        if (h.x < 0.0003f * t + 0.0005f) { mat = int(h.y); break; }
        t += h.x * 0.8f;                                   // several SDFs are bounds, not exact
        if (t > tMax) break;
    }
    if (mat == MAT_NONE && flatHit) { mat = MAT_TERRAIN; t = tPlane; }

    // Close-up grass canopy: real blades in front of whatever the main march hit.
    {
        float hFrac; float2 bladeId;
        const float tg = traceGrass(ro, rd, eye, (mat == MAT_NONE) ? INFINITY : t, hFrac, bladeId);
        if (tg > 0.0f) return shadeGrassBlade(ro + rd * tg, rd, eye, hFrac, bladeId);
    }

    if (mat == MAT_NONE) return environment(rd);

    const float3 pos = ro + rd * t;

    // Normals: analytic for the ball and terrain (most of the frame), SDF
    // gradient for everything else.
    float3 n;
    if (mat == MAT_BALL) {
        n = normalize(pos - kBallCenter);
    } else if (mat == MAT_TERRAIN) {
        const float slope = (terrainHeight(pos.z + 1.0f) - terrainHeight(pos.z - 1.0f)) * 0.5f;
        n = normalize(float3(0.0f, 1.0f, -slope));
    } else {
        n = calcNormal(pos, 0.0005f + 0.0003f * t);
    }

    // Material parameters.
    Surface s;
    float   aoBoost = 1.0f;
    switch (mat) {
        case MAT_BALL: {
            const float pixelAngle = 2.0f / (resolution.y * focal);
            const float footprint  = t * pixelAngle * subpixelScale / kBallRadius / max(dot(n, -rd), 0.15f);
            const float4 dn = dimpleNormal(n, float(u.dimpleCount), footprint);
            n       = dn.xyz;
            aoBoost = dn.w;
            s = dielectric(float3(0.90f, 0.90f, 0.88f), 0.35f);          // urethane cover
            break;
        }
        case MAT_TEE: {
            const float grain = valueNoise(float2(atan2(pos.z, pos.x) * 3.0f, pos.y * 14.0f));
            float3 albedo = mix(float3(0.80f, 0.64f, 0.43f), float3(0.66f, 0.49f, 0.31f), grain);
            albedo *= mix(0.55f, 1.0f, smoothstep(0.0f, 0.45f, pos.y));
            s = dielectric(albedo, 0.65f);
            break;
        }
        case MAT_HEAD:      s = clubHeadSurface(pos, n); break;
        case MAT_FERRULE:   s = dielectric(float3(0.015f), 0.25f); break;
        case MAT_SHAFT:     s = metallic(float3(0.80f, 0.80f, 0.81f), 0.06f); break;          // chrome steel
        case MAT_GRIP:      s = dielectric(float3(0.03f), 0.75f); break;                   // rubber

        case MAT_SHIRT: {                                                                    // Sunday red mock
            const float3 t = toSpine(pos);
            const bool logo = t.z > 4.0f && sdSwoosh((t.xy - float2(9.0f, 37.0f)) / 3.0f) < 0.0f;
            s = logo ? dielectric(float3(0.80f), 0.6f) : dielectric(float3(0.50f, 0.028f, 0.032f), 0.75f, 0.6f);
            break;
        }
        case MAT_PANTS:     s = dielectric(float3(0.018f), 0.70f, 0.5f); break;
        case MAT_BELT:      s = dielectric(float3(0.02f), 0.35f); break;                   // leather
        case MAT_SKIN: {
            // Medium-brown skin with darker lips; the eyes get sclera and a dark
            // iris gazing down at the ball.
            const float3 hl = toHead(pos);
            const float3 eo = float3(abs(hl.x), hl.y, hl.z) - kEyeL;
            if (length(eo) < kEyeRadius + 0.05f && hl.z > 8.0f) {
                const bool iris = dot(normalize(eo), float3(0.0f, -0.55f, 0.835f)) > 0.93f;
                s = iris ? dielectric(float3(0.02f, 0.012f, 0.008f), 0.1f) : dielectric(float3(0.55f, 0.50f, 0.45f), 0.15f);
            } else {
                const float2 lq = float2(hl.x / 2.5f, (hl.y + 5.55f) / 1.35f);
                const float  lips = (1.0f - smoothstep(0.75f, 1.0f, length(lq))) * step(8.6f, hl.z);
                s = dielectric(mix(float3(0.25f, 0.14f, 0.085f), float3(0.19f, 0.085f, 0.065f), lips), 0.55f, 0.0f, 1.0f);
            }
            break;
        }
        case MAT_GLOVE:     s = dielectric(float3(0.85f, 0.85f, 0.82f), 0.80f, 0.4f); break; // cabretta
        case MAT_CAP: {                                                                      // black six-panel cap
            const float3 hl   = toHead(pos);
            const float  pan  = abs(fract(atan2(hl.x, hl.z) * (3.0f / M_PI_F) + 0.5f) - 0.5f) * length(hl.xz) * (M_PI_F / 3.0f);
            const bool   seam = hl.y > 3.0f && pan < 0.10f;
            const bool   logo = hl.z > 5.0f && hl.y > 3.0f && sdSwoosh((hl.xy - float2(0.0f, 6.0f)) / 3.3f) < 0.0f;
            s = logo ? dielectric(float3(0.80f), 0.6f) : dielectric(float3(seam ? 0.01f : 0.02f), 0.85f, 0.6f);
            break;
        }
        case MAT_SHOE:      s = (pos.y < 2.0f) ? dielectric(float3(0.80f), 0.6f)
                                               : dielectric(float3(0.02f), 0.25f); break;

        case MAT_BAG: {
            // White body, black base and cuff, red pinstripe, black script on the front.
            const float3 b   = toBag(pos);
            const float  ang = atan2(b.x, b.z);
            float3 albedo = float3(0.84f, 0.84f, 0.83f);
            if (b.y < 14.0f || b.y > kBagHeight - 6.0f) albedo = float3(0.025f);
            if (abs(b.y - 15.5f) < 0.9f)                albedo = float3(0.55f, 0.03f, 0.04f);
            const float capH  = 10.0f;
            const float width = textWidth(kTextTitleist, 8, 0.02f) * capH;
            const float txt   = sdText((float2(ang * kBagRadius, b.y) - float2(-0.5f * width, 56.0f)) / capH,
                                       kTextTitleist, 8, 0.02f, 0.28f) * capH;
            if (abs(ang) < 1.4f && txt < 0.75f) albedo = float3(0.02f);
            s = dielectric(albedo, 0.45f);
            break;
        }
        case MAT_HEADCOVER: {
            const bool stripe = abs(toBag(pos).y - (kBagHeight + 9.0f)) < 1.2f;
            s = dielectric(stripe ? float3(0.55f, 0.03f, 0.04f) : float3(0.03f), 0.9f, 0.5f);
            break;
        }
        case MAT_BAG_IRON:  s = metallic(float3(0.64f, 0.65f, 0.67f), 0.2f); break;
        case MAT_TOWEL: {                                                              // terry cloth, woven bands
            const float y = toBag(pos).y - (kBagHeight - 28.0f);
            const float3 band = (y > 0.0f && y < 1.6f) ? float3(0.55f, 0.03f, 0.04f)
                              : ((y > 2.2f && y < 3.0f) ? float3(0.03f) : float3(0.78f, 0.77f, 0.73f));
            s = dielectric(band, 1.0f, 0.8f);
            break;
        }
        case MAT_MARKER:    s = dielectric(float3(0.03f, 0.13f, 0.06f), 0.45f); break;       // Augusta green
        case MAT_BOARD: {
            // Hand-operated leaderboard: green board, white header, white name
            // plates and score cells with dark names and red under-par numbers.
            const float2 q = pos.xy - kBoardMin.xy;
            const float2 size = kBoardMax.xy - kBoardMin.xy;
            float3 c = float3(0.03f, 0.16f, 0.07f);
            if (pos.z > kBoardMax.z - 0.5f && q.y > 0.0f) {
                const float row  = floor((size.y - 70.0f - q.y) / 30.0f);
                const float rv   = fract((size.y - 70.0f - q.y) / 30.0f);
                if (q.y > size.y - 60.0f && q.y < size.y - 12.0f && abs(q.x - 0.5f * size.x) < 220.0f) {
                    c = float3(0.85f);                                                  // header plate
                    const float2 g = float2(fract(q.x / 30.0f), (q.y - size.y + 52.0f) / 32.0f);
                    if (g.y > 0.0f && g.y < 1.0f && g.x > 0.2f && g.x < 0.8f && hash12(floor(q.xy / float2(30.0f, 8.0f))) > 0.35f)
                        c = float3(0.03f, 0.16f, 0.07f);                              // lettering
                } else if (row >= 0.0f && row < 11.0f && rv > 0.12f && rv < 0.88f) {
                    const bool name = q.x > 20.0f && q.x < 330.0f;
                    const bool cell = q.x > 350.0f && q.x < size.x - 20.0f && fract((q.x - 350.0f) / 28.0f) < 0.82f;
                    if (name || cell) {
                        c = float3(0.85f);
                        const float glyph = hash12(floor(float2(q.x / 9.0f, row)));
                        const bool  ink   = rv > 0.28f && rv < 0.72f && fract(q.x / 9.0f) < 0.7f;
                        if (name && ink && glyph > 0.25f && q.x < 120.0f + 200.0f * hash12(float2(row, 3.0f)))
                            c = float3(0.03f);
                        if (cell && ink && fract((q.x - 350.0f) / 28.0f) > 0.2f && fract((q.x - 350.0f) / 28.0f) < 0.62f
                            && hash12(float2(floor((q.x - 350.0f) / 28.0f), row)) < 0.8f - 0.05f * row)
                            c = (hash12(float2(row, 9.0f)) < 0.85f) ? float3(0.60f, 0.03f, 0.03f) : float3(0.03f, 0.16f, 0.07f);
                    }
                }
            }
            s = dielectric(c, 0.6f);
            break;
        }

        case MAT_FAN_SKIN:  s = dielectric(fanColor(pos, mat), 0.5f, 0.0f, 1.0f); break;
        case MAT_FAN_SHIRT:
        case MAT_FAN_PANTS:
        case MAT_FAN_HAT:   s = dielectric(fanColor(pos, mat), 0.85f, 0.3f); break;
        case MAT_ROPE:      s = dielectric(float3(0.10f, 0.28f, 0.14f), 0.8f); break;
        case MAT_TRUNK: {
            const float bark = valueNoise(float2(atan2(n.z, n.x) * 4.0f, pos.y * 0.02f));
            s = dielectric(mix(float3(0.16f, 0.11f, 0.08f), float3(0.30f, 0.22f, 0.16f), bark), 0.95f);
            break;
        }
        case MAT_CANOPY: {
            const float v = valueNoise(pos.xy * 0.02f + pos.z * 0.01f);
            s = dielectric(mix(float3(0.03f, 0.09f, 0.03f), float3(0.08f, 0.17f, 0.05f), v), 0.8f, 0.0f, 0.6f);
            break;
        }
        case MAT_AZALEA: {
            const float h      = hash12(float2(round(pos.x / 260.0f), 3.0f));
            const float3 bloom = (h < 0.45f) ? float3(0.85f, 0.18f, 0.45f)
                               : (h < 0.8f)  ? float3(0.92f, 0.45f, 0.62f) : float3(0.90f, 0.88f, 0.86f);
            const float leaf   = step(0.72f, valueNoise(pos.xy * 0.15f + pos.z * 0.07f));
            s = dielectric(mix(bloom, float3(0.06f, 0.16f, 0.05f), leaf), 0.7f, 0.2f, 0.5f);
            break;
        }
        default:
        {
            // Under the blades only thatch shows; just beyond them the flat turf
            // takes on the darker, self-shadowed look of a blade canopy seen low,
            // so the geometry-to-texture hand-off doesn't show.
            const float dEye = distance(pos.xz, eye.xz);
            s = turf(pos, t, rd, n);
            s.albedo *= mix(mix(0.42f, 1.0f, smoothstep(160.0f, 700.0f, dEye)), 0.45f, grassFade(pos.xz, eye));
            break;
        }
    }

    // Lighting. Surfaces turned well away from the sun are in their own shadow,
    // so the 128-step shadow march is skipped for them.
    const bool  isProp   = mat <= MAT_GRIP;
    const bool  isPlayer = mat >= MAT_SHIRT && mat <= MAT_SHOE;
    const float aoScale  = isProp ? 1.2f : (isPlayer ? 10.0f : (length(pos.xz) < 400.0f ? 12.0f : 60.0f));
    const float3 L = normalize(kSunDir);

    Light li;
    li.L      = L;
    li.shadow = (dot(n, L) > -0.3f)
              ? softShadow(pos + n * (0.01f + 0.0005f * t), L, 12.0f, 0.02f + 0.001f * t) : 0.0f;
    li.ao     = ambientOcclusion(pos, n, aoScale) * aoBoost;
    const float3 col = shade(s, n, rd, li);

    // Aerial perspective into the pines.
    const float haze = 1.0f - exp(-t * 0.00007f);
    return mix(col, kSkyHorizon * 0.95f, haze);
}

// MARK: - Tonemapping -----------------------------------------------------------

static float3 tonemapACES(float3 x)
{
    return saturate((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f));
}

// MARK: - Kernel ----------------------------------------------------------------

kernel void golfSceneKernel(texture2d<half, access::write> outTexture [[texture(0)]],
                            constant GolfSceneUniforms&    u          [[buffer(0)]],
                            constant uint2&                tileOrigin [[buffer(1)]],
                            uint2                          gid        [[thread_position_in_grid]],
                            uint2                          tid        [[thread_position_in_threadgroup]],
                            uint2                          tgid       [[threadgroup_position_in_grid]],
                            uint2                          tgSize     [[threads_per_threadgroup]])
{
    // This dispatch covers one band. Map the thread to its pixel and bounds-check.
    const uint2 pixel  = tileOrigin + gid;
    const uint  width  = outTexture.get_width();
    const uint  height = outTexture.get_height();
    if (pixel.x >= width || pixel.y >= height) {
        return;
    }
    const float2 resolution = float2(width, height);

    // NxN stratified pixel samples, each paired with a point on the lens disk
    // (golden-angle spiral, rotated per pixel to turn banding into fine noise).
    const uint  spp   = clamp(u.samplesPerAxis, 1u, 6u);
    const uint  count = spp * spp;
    const float inv   = 1.0f / float(spp);
    const float rot   = hash12(float2(pixel)) * 2.0f * M_PI_F;
    float3      color = float3(0.0f);
    for (uint i = 0; i < count; ++i) {
        const float2 offset = (float2(i % spp, i / spp) + 0.5f) * inv;
        const float  r      = sqrt((float(i) + 0.5f) / float(count));
        const float  a      = float(i) * 2.39996323f + rot;
        color += renderSample(float2(pixel) + offset, resolution, r * float2(cos(a), sin(a)), u, inv);
    }
    color /= float(count);

    // Exposure, tonemap, vignette, gamma (target is linear .bgra8Unorm).
    color = tonemapACES(color * 0.8f);
    const float2 v = float2(pixel) / resolution - 0.5f;
    color *= 1.0f - 0.35f * dot(v, v);
    color = pow(color, float3(1.0f / 2.2f));

    if (u.debugTiles != 0u) {
        const bool odd = ((tgid.x + tgid.y) & 1u) != 0u;
        color *= odd ? float3(1.0f, 0.55f, 0.55f) : float3(0.55f, 0.75f, 1.0f);
        const bool border = tid.x == 0u || tid.y == 0u ||
                            tid.x == tgSize.x - 1u || tid.y == tgSize.y - 1u;
        if (border) color = mix(color, float3(1.0f, 1.0f, 0.0f), 0.6f);
    }

    outTexture.write(half4(half3(color), 1.0h), pixel);
}
