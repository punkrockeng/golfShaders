//
//  GolfBallApp.swift
//
//  App entry point: one window showing the Metal-rendered spinning golf ball.
//

import SwiftUI

@main
struct GolfBallApp: App {
    var body: some Scene {
        WindowGroup("Golf Ball") {
            // 1024² render target, displayed sharp on Retina at the default window size.
            GolfBallView(width: 1024, height: 1024)
                .frame(minWidth: 400, minHeight: 400)
        }
        .defaultSize(width: 720, height: 720)
    }
}
