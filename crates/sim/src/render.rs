//! Offscreen MuJoCo rendering for demo frames.

use std::path::Path;

use mujoco_rs::renderer::MjRenderer;
use mujoco_rs::wrappers::MjvCamera;

use crate::{FlyWorld, SimError};

/// A reusable offscreen renderer tied to one fly world.
pub struct FrameRenderer {
    renderer: MjRenderer,
}

impl FlyWorld {
    /// Create a renderer that looks at the tethered fly and the ball.
    pub fn frame_renderer(&self, width: usize, height: usize) -> Result<FrameRenderer, SimError> {
        let model = self.renderer_model();
        let mut camera = MjvCamera::new_free(model.as_ref());
        camera.lookat = [0.0, 0.0, -0.05];
        camera.distance = 1.05;
        camera.azimuth = 135.0;
        camera.elevation = -12.0;

        let mut renderer = MjRenderer::builder()
            .width(width as u32)
            .height(height as u32)
            .num_visual_user_geom(0)
            .rgb(true)
            .depth(false)
            .build(model)
            .map_err(|error| SimError::Data(format!("could not create renderer: {error}")))?;
        renderer.set_camera(camera);
        Ok(FrameRenderer { renderer })
    }

    /// Render the current state to a PNG file.
    pub fn render_frame(
        &mut self,
        renderer: &mut FrameRenderer,
        path: impl AsRef<Path>,
    ) -> Result<(), SimError> {
        renderer
            .renderer
            .sync_data(self.renderer_data())
            .map_err(|error| SimError::Data(format!("could not sync renderer: {error}")))?;
        renderer
            .renderer
            .render()
            .map_err(|error| SimError::Data(format!("could not render frame: {error}")))?;
        renderer
            .renderer
            .save_rgb(path)
            .map_err(|error| SimError::Data(format!("could not save frame: {error}")))
    }
}
