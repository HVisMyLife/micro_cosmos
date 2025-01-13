use bevy::{prelude::*, diagnostic::{Diagnostics, FrameTimeDiagnosticsPlugin}, input::mouse::{MouseWheel, MouseScrollUnit}, render::camera::RenderTarget};
use bevy_egui::{egui::{self, ColorImage}, EguiContext, EguiPlugin};
use bevy_egui::egui::plot::{PlotPoints, Line, Plot};
use bevy_pixel_camera::PixelCameraPlugin;

use rusty_neat::{NN, ActFunc};
use simplesvg as svg;

use bevy_egui::EguiSettings;

fn update_ui_scale_factor(mut egui_settings: ResMut<EguiSettings>, windows: Res<Windows>) {
    if let Some(window) = windows.get_primary() {
        egui_settings.scale_factor = 1.2 / window.scale_factor();
    }
}

#[derive(Resource, Default, Debug)]
pub struct MinionAmount(pub usize);

#[derive(Debug, Clone, Resource)]
pub struct SelectedNN{
    pub nn: NN,
    pub pos: Vec2,
    pub hp: f32,
    pub hunger: f32,
    pub age: f32,
    pub eid: Entity,
    pub eid_old: Entity
}
impl Default for SelectedNN{
    fn default() -> Self { 
        let mut n = NN::new(1, 1); 
        n.forward(&[0.5]); 
        Self { 
            nn: n,
            pos: Vec2::default(),
            hp: 0.0,
            hunger: 0.0,
            age: 0.0,
            eid: Entity::from_raw(0),
            eid_old: Entity::from_raw(1)
        } 
    }
}

#[derive(Default, Debug, Resource)]
struct PlotPop{
    pp: Vec<[f64; 2]>
}

#[derive(Resource)]
struct PlotTimer(Timer);


// drawing plot of alive entities, when full eliminate odd/even logs, 
// so older means less resolution
fn u_plot(
    time: Res<Time>, 
    mut timer: ResMut<PlotTimer>,
    m_a: Res<MinionAmount>,
    mut plot_p: ResMut<PlotPop>
){
    if timer.0.tick(time.delta()).just_finished() { 
        let idx = plot_p.pp.last().unwrap()[0] + 1.0;
        plot_p.pp.push([idx, m_a.0 as f64]);
        if plot_p.pp.len() > 200 {
            let np: Vec<[f64; 2]> = plot_p.pp.iter().enumerate().filter(|f| f.0 % 2 == 1).map(|f| *f.1).collect::<_>();
            plot_p.pp = np;
        }
    }

}

#[derive(Default, Resource)]
struct ImageData {
    egui_texture_handle: Option<egui::TextureHandle>,
}


// generating svg of currently selected entity
fn svg_nn(nn: &NN) -> (u32, u32, Vec<u8>) {
    let mut objs: Vec<svg::Fig> = vec![];
    let mut positions: Vec<(f32, f32)> = vec![(0_f32, 0_f32); nn.nodes.len()];
    
    nn.layer_order.iter().enumerate().for_each(|(x, l)|{
        l.iter().enumerate().for_each(|(y, p)|{
            positions[*p] = ((x + 1) as f32 * 64.0, (y + 1) as f32 * 64.0);

            let mut cir = svg::Fig::Circle(positions[*p].0, positions[*p].1, 16.0);
            let mut att = svg::Attr::default();
            att = svg::Attr::fill(att, svg::ColorAttr::Color(
                (nn.nodes[*p].bias * 255.0).max(0.0) as u8, 
                0, 
                (nn.nodes[*p].bias * -255.0).max(0.0) as u8));
            cir = cir.styled(att);
            objs.push(cir);
            
            let mut cir = svg::Fig::Circle(positions[*p].0, positions[*p].1, 8.0);
            let mut att = svg::Attr::default();
            match nn.nodes[*p].act_func {
                ActFunc::None => att = svg::Attr::fill(att, svg::ColorAttr::Color(0, 0, 0)),
                ActFunc::Tanh => att = svg::Attr::fill(att, svg::ColorAttr::Color(255, 255, 255)),
                ActFunc::ReLU => att = svg::Attr::fill(att, svg::ColorAttr::Color(153, 0, 153)),
                ActFunc::Sigmoid => att = svg::Attr::fill(att, svg::ColorAttr::Color(0, 204, 0)),
            }
        
            cir = cir.styled(att);
            objs.push(cir);
        });
    });

    nn.connections.iter().filter(|e| e.active).for_each(|c| {
        let mut lin = svg::Fig::Line(
            positions[c.from].0,
            positions[c.from].1,
            positions[c.to].0,
            positions[c.to].1,
        );
        let mut att = svg::Attr::default();
        att = svg::Attr::stroke(att, svg::ColorAttr::Color(
            (c.weight > 0.0) as u8 * 255, 
            0, 
            (c.weight < 0.0) as u8 * 255));
        att = svg::Attr::stroke_width(att, (c.weight * 4.0).abs() as f32);
        lin = lin.styled(att);
        objs.push(lin);
    });


    let out = svg::Svg{0: objs, 1: 720, 2: 640};
    //println!("{}", out.to_string());
    let svg = nsvg::parse_str(&out.to_string(), nsvg::Units::Pixel, 96.0).unwrap();
    svg.rasterize_to_raw_rgba(0.5).unwrap()
}

// create image from svg
fn u_img(
    mut h_texture: ResMut<ImageData>,
    mut selected: ResMut<SelectedNN>,
){    
    if selected.eid != selected.eid_old { 
        selected.eid_old = selected.eid;
        let (size_x, size_y, pxs) = svg_nn(&selected.nn);
        let img = ColorImage::from_rgba_unmultiplied([size_x as usize, size_y as usize], &pxs);
        if let Some(x) = h_texture.egui_texture_handle.as_mut(){
            x.set(img, default());
        }
    }
}

// main gui creator
fn ui_window(
    time: Res<Time>, 
    sel: Res<SelectedNN>,
    mut h_texture: ResMut<ImageData>,
    mut egui_ctx: ResMut<EguiContext>,
    diagnostics: Res<Diagnostics>, 
    plot_p: Res<PlotPop>,
    w_p: Res<CursorWorld>,
) {
    let texture = h_texture
        .egui_texture_handle
        .get_or_insert_with(|| {
            let mut n = NN::new(1, 1); 
            n.forward(&[0.5]); 
            let (size_x, size_y, pxs) = svg_nn(&n);
            let img = ColorImage::from_rgba_unmultiplied([size_x as usize, size_y as usize], &pxs);
            egui_ctx.ctx_mut().load_texture(
                "nn",
                img.clone(),
                Default::default(),
            )
        })
        .clone();

    egui::Window::new("---").show(egui_ctx.ctx_mut(), |ui|{
        ui.heading("Params");
        let l_res = diagnostics.get(FrameTimeDiagnosticsPlugin::FPS);
        if let Some(v) = l_res {
            let v_sm = v.smoothed().unwrap_or(0.0);
            ui.label(format!("Fps: \n{v_sm:.0}"));
        }

        let v = time.elapsed_seconds();
        ui.label(format!("Elapsed: \n{v:.0}s"));

        ui.label(format!("Camera mode: "));
        if w_p.follow {
            ui.label(format!("- follow minion"));
        } else {
            ui.label(format!("- cursor"));
        }

        ui.separator();
        
        ui.heading("Best");
        ui.label(format!("Age: {:.0}", sel.age));
        ui.label(format!("Generation: {:.0}", sel.nn.generation));
        ui.label("HP:");
        ui.add(egui::widgets::ProgressBar::new(sel.hp));
        ui.label("Hunger:");
        ui.add(egui::widgets::ProgressBar::new(sel.hunger/2.0));

        ui.add(egui::widgets::Image::new(
            texture.id(),
            texture.size_vec2(),    
        ));
        

        ui.separator();
        ui.heading("Population size:");

        Plot::new("Population").show(ui, |plot_ui|{
            let plot: PlotPoints = plot_p.pp.clone().into();
            let line = Line::new(plot);
            plot_ui.line(line);
        });

        

    });
}


pub struct UiManPlugin;
impl Plugin for UiManPlugin {
    fn build(&self, app: &mut App){
        app
            .add_plugin(EguiPlugin)
            .add_plugin(PixelCameraPlugin)
            .add_system(update_ui_scale_factor)
            .add_startup_system(setup_ui)
            .add_system(update_cam)
            .add_system(cursor_system)
            .add_system(ui_window)
            .init_resource::<ImageData>()
            .add_system(u_plot)
            .add_system(u_img)
            .insert_resource(PlotTimer(Timer::from_seconds(2.0, TimerMode::Repeating)))
            .insert_resource(MinionAmount(0))
            .insert_resource(SelectedNN::default())
            .insert_resource(PlotPop {pp: vec![[0.0, 0.0]]})
            .insert_resource(CursorWorld::default())
        ;
    }
}

#[derive(Component)]
struct MainCamera;

#[derive(Default, Debug, Resource)]
pub struct CursorWorld{
    pub x: f32,
    pub y: f32,
    pub follow: bool
}

// controlling camera, eg either following best minion
// or moving with mouse scrolls (like in fusion360)
fn update_cam( 
    time: Res<Time>,
    mut query: Query<&mut Transform, With<Camera2d>>, 
    keyboard_input: Res<Input<KeyCode>>,
    mut scroll_evr: EventReader<MouseWheel>,
    //mut motion_evr: EventReader<MouseMotion>,
    //buttons: Res<Input<MouseButton>>,
    mut w_p: ResMut<CursorWorld>,
    sel: Res<SelectedNN>,
) {
    let mut cam = query.get_single_mut().unwrap();
    let mut mov = Vec3::new(0.0, 0.0, 0.0);
    let mut sc = 0_f32;

    for ev in scroll_evr.iter() {
        match ev.unit {
            MouseScrollUnit::Line => {
                sc = ev.y / 10.0;
                mov = Vec3::new(cam.translation.x - w_p.x, cam.translation.y - w_p.y, 0.0) * ev.y / 10.0;
            }
            MouseScrollUnit::Pixel => {println!("ERR: Mouse Scroll");}
        }
    }
    let c = cam.scale;
    if keyboard_input.just_pressed(KeyCode::Space) {w_p.follow = !w_p.follow;}
    if w_p.follow {
        let v = cam.translation.truncate().lerp(sel.pos, time.delta_seconds()*3.0);
        cam.translation.x = v.x;
        cam.translation.y = v.y;
    }
    else {
        cam.translation += mov;
    }
    cam.translation = cam.translation.clamp(Vec3::new(-5000.0, -5000.0, -10.0), Vec3::new(5000.0, 5000.0, 10.0));
    cam.scale += c * Vec3::new(sc, sc, 0.0);
}

// update world mouse position
fn cursor_system(
    // need to get window dimensions
    wnds: Res<Windows>,
    // query to get camera transform
    q_camera: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    mut w_p: ResMut<CursorWorld>
) {
    let (camera, camera_transform) = q_camera.single();

    // get the window that the camera is displaying to (or the primary window)
    let wnd = if let RenderTarget::Window(id) = camera.target {
        wnds.get(id).unwrap()
    } else { wnds.get_primary().unwrap() };

    if let Some(screen_pos) = wnd.cursor_position() {
        // get the size of the window
        let window_size = Vec2::new(wnd.width() as f32, wnd.height() as f32);
        // convert screen position [0..resolution] to ndc [-1..1] (gpu coordinates)
        let ndc = (screen_pos / window_size) * 2.0 - Vec2::ONE;
        // matrix for undoing the projection and camera transform
        let ndc_to_world = camera_transform.compute_matrix() * camera.projection_matrix().inverse();
        // use it to convert ndc to world-space coordinates
        let world_pos = ndc_to_world.project_point3(ndc.extend(-1.0));
        // reduce it to a 2D value
        let world_pos: Vec2 = world_pos.truncate();
        w_p.x = world_pos.x;
        w_p.y = world_pos.y;
    }
}


fn setup_ui(mut commands: Commands) {
    //commands.spawn(Camera2dBundle::default());
    commands.spawn(Camera2dBundle::default())
        .insert(MainCamera)
    ;
}






//fn draw_nn(nn: &NN) {
//    for i in 0..nn.connections.len(){ if nn.connections[i].active {
//        draw_line(
//            nn.positions[nn.connections[i].from].0, 
//            nn.positions[nn.connections[i].from].1, 
//            nn.positions[nn.connections[i].to].0, 
//            nn.positions[nn.connections[i].to].1, 
//            (nn.connections[i].weight * 8.0).abs() as f32, 
//            Color::from_rgba((nn.connections[i].weight > 0.0) as u8 * 255, 0, (nn.connections[i].weight < 0.0) as u8 * 255, 128)
//        );
//        draw_line(
//            nn.positions[nn.connections[i].from].0 + ((nn.positions[nn.connections[i].to].0 - nn.positions[nn.connections[i].from].0) * 1.1), 
//            nn.positions[nn.connections[i].from].1 + ((nn.positions[nn.connections[i].to].1 - nn.positions[nn.connections[i].from].1) * 1.1), 
//            nn.positions[nn.connections[i].to].0, 
//            nn.positions[nn.connections[i].to].1, 
//            4.0, 
//            Color::from_rgba(0, 0, 0, 192)
//        );
//    }}
//    for i in 0..nn.nodes.len() {
//        draw_circle(
//            nn.positions[i].0, nn.positions[i].1, 
//            16.0, 
//            Color::from_rgba((nn.nodes[i].bias * 255.0).max(0.0) as u8, 0, (nn.nodes[i].bias * -255.0).max(0.0) as u8, 96)
//        );
//        draw_circle_lines(
//            nn.positions[i].0, nn.positions[i].1, 
//            16.0, 
//            2.0,
//            Color::from_rgba(0, 0, 0, 192)
//        );
//        draw_text(
//            &nn.positions[i].2.to_string(), 
//            nn.positions[i].0-8.0, nn.positions[i].1+8.0, 
//            32.0, 
//            Color::from_rgba(0, 0, 0, 255)
//        )
//
//    }
//}


//
//#[derive(Component)]
//struct FpsText;
//
//#[derive(Component)]
//struct TimeText;
//
//fn setup_ui(mut commands: Commands, asset_server: Res<AssetServer>) {
//    commands.spawn(Camera2dBundle::default());
//   
//    commands.spawn(FpsText)
//        .insert(text(&asset_server, "Fps: ", UiRect::new(Val::Auto, Val::Percent(50.0), Val::Percent(3.0), Val::Auto) ))
//    ;
//
//    commands.spawn(TimeText)
//        .insert(text(&asset_server, "Time: ", UiRect::new(Val::Auto, Val::Percent(50.0), Val::Percent(6.0), Val::Auto) ))
//    ;
//
//}
//
//fn text(asset_server: &Res<AssetServer>, label: &str, pos: UiRect) -> TextBundle {
//    TextBundle {
//        style: Style {
//            position: pos,
//            ..Default::default()
//        },
//        text: Text::from_sections([
//            TextSection::new(
//                label,
//                TextStyle {
//                    font: asset_server.load("fonts/FiraMono-Bold.ttf"),
//                    font_size: 50.0,
//                    color: Color::rgba(1.0, 0.8, 0.0, 0.1),
//                },
//            ),
//            TextSection::from_style(TextStyle {
//                font: asset_server.load("fonts/FiraMono-Medium.ttf"),
//                font_size: 50.0,
//                color: Color::rgba(1.0, 0.3, 0.8, 0.1),
//            }),
//        ]),
//        ..Default::default()
//    }
//}
//
//fn update_ui_fps(diagnostics: Res<Diagnostics>, mut query: Query<&mut Text, With<FpsText>>) {
//    for mut text in &mut query {
//        if let Some(fps) = diagnostics.get(FrameTimeDiagnosticsPlugin::FPS) {
//            if let Some(value) = fps.smoothed() {
//                // Update the value of the second section
//                text.sections[1].value = format!("{value:.2}");
//            }
//        }
//    }
//}
//
//fn update_ui_time(time: Res<Time>, mut query: Query<&mut Text, With<TimeText>>) {
//    for mut text in &mut query {
//        let value = time.elapsed_seconds();
//        text.sections[1].value = format!("{value:.1}");
//    }
//}



