#![allow(clippy::type_complexity)]

use std::f32::consts::PI;

//use rayon::prelude::*;
use serde::{Serialize, Deserialize};
//use bincode::{serialize, deserialize};
use rusty_neat::*;
use bevy::{prelude::*, diagnostic::FrameTimeDiagnosticsPlugin};
use bevy_rapier2d::prelude::*;
use fastrand as fr;

mod ui;
use ui::*;


#[derive(Debug, Clone, Serialize, Deserialize, Component)]
struct Minion;

#[derive(Debug, Clone, Serialize, Deserialize, Component)]
struct Control;

#[derive(Debug, Clone, Serialize, Deserialize, Component)]
struct Cursor;

#[derive(Debug, Clone, Serialize, Deserialize, Component)]
struct Detector;

#[derive(Debug, Clone, Serialize, Deserialize, Component)]
struct Food;


#[derive(Debug, Clone, Serialize, Deserialize, Resource)]
struct BestNN{
    nn: NN,
    age: f32
}
impl Default for BestNN{
    fn default() -> Self { Self { nn: NN::new(0, 0), age: 0.0 } }
}

#[derive(Debug, Clone, Serialize, Deserialize, Component)]
struct Brain{
    nn: NN, 
    sight_minions: Vec<(f32, f32)>, // angles and distances to minions in sight
    sight_food: Vec<(f32, f32)>, // angles and distances to food in sight
    //attacking: bool,
    //eating: bool
}
impl Default for Brain {
    fn default() -> Self {
        let mut n = NN::new(8, 2); 
        n.forward(&[0.5]); 
        Self { 
        nn: n, 
        sight_minions: vec![(0.0, f32::MAX)], 
        sight_food: vec![(0.0, f32::MAX)], 
        //attacking: false,
        //eating: false
    }}
}

#[derive(Debug, Clone, Serialize, Deserialize, Component)]
struct Sight{radius: f32}
impl Default for Sight {
    fn default() -> Self {Self { radius: 400.0 }}
}

#[derive(Debug, Clone, Serialize, Deserialize, Component)]
struct Size{radius: f32}
impl Default for Size {
    fn default() -> Self {Self { radius: 16.0 }}
}

#[derive(Debug, Clone, Serialize, Deserialize, Component)]
struct Health{health: f32, protection: f32}
impl Health {
    fn new(hp: f32) -> Self { Self { health: hp, protection: 1.0 } }
}
impl Default for Health {
    fn default() -> Self {Self { health: 1.0, protection: 1.0 }}
}

#[derive(Debug, Clone, Serialize, Deserialize, Component)]
struct Weapon{damage: f32, width: f32}
impl Default for Weapon {
    fn default() -> Self {Self { damage: 0.2, width: 16.0 }}
}

#[derive(Debug, Clone, Serialize, Deserialize, Component)]
struct Age{age: f32, lifespan: f32}
impl Default for Age{
    fn default() -> Self {Self{age: 0.0, lifespan: 720.0}}
}

#[derive(Debug, Clone, Serialize, Deserialize, Component)]
struct Hunger{filled: f32, metabolism: f32}
impl Default for Hunger{
    fn default() -> Self {Self{filled: 1.0, metabolism: 0.02}}
}



fn ai( mut m_a: ResMut<MinionAmount>,
    mut query: Query<(&mut Brain, &mut ExternalForce, &mut Transform, &mut Velocity, &Hunger, &Health), (With<Minion>, Without<Control>)>,
) {
    m_a.0 = query.iter().len();
    query.par_for_each_mut(16, |(mut brain, mut force, mut transform, velocity, hunger, hp)| {
        let mut s_minions = (0_f32, f32::MAX);
        brain.sight_minions.iter().for_each(|s|{
            if s.1 < s_minions.1 { s_minions = *s}
        });
        brain.sight_minions.clear();
        //let atc = brain.attacking as i32 as f64;
        //brain.attacking = false;

        let mut s_foods = (0_f32, f32::MAX);
        brain.sight_food.iter().for_each(|s|{
            if s.1 < s_foods.1 { s_foods = *s}
        });
        brain.sight_food.clear();
        //let eat = brain.eating as i32 as f64;
        //brain.eating = false;


        let out = brain.nn.forward(&[
            velocity.linvel.length() as f64,
            velocity.angvel as f64,
            hp.health as f64,
            hunger.filled as f64,
            s_minions.0 as f64,
            1.0/s_minions.1 as f64,
            s_foods.0 as f64,
            1.0/s_foods.1 as f64,
            ]);

        force.force = Vec2::new(0.0, (out[0] * 20.0).clamp(-20.0, 20.0) as f32);
        force.torque = (out[1]/100.0).clamp(-0.01, 0.01) as f32;

        let dir = Vec2::new( transform.local_x().x, transform.local_x().y);
        force.force = force.force.rotate(dir);

        if transform.translation.x >  7_000.0 {transform.translation.x = -7_000.0}
        if transform.translation.x < -7_000.0 {transform.translation.x =  7_000.0}
        if transform.translation.y >  7_000.0 {transform.translation.y = -7_000.0}
        if transform.translation.y < -7_000.0 {transform.translation.y =  7_000.0}
    });
}

fn attack(
    time: Res<Time>,
    rapier_context: Res<RapierContext>, 
    mut q_detector: Query<(&Parent, &Weapon, &Collider)>,
    mut q_minions: Query<(&mut Brain, &mut Health, &mut Hunger), With<Minion>>,
    mut q_food: Query<&mut Health, (Without<Minion>, With<Food>)>
){
    rapier_context.intersection_pairs().filter(|pp| pp.2 ).for_each(|pp|{
        let mut p = (pp.0, pp.1);

        if q_detector.contains(pp.1) {
            p = (pp.1, pp.0);
        }

        if q_detector.contains(p.0) && q_minions.contains(p.1) {
            let child = q_detector.get_mut(p.0).unwrap();
            let res = q_minions.get_many_mut([child.0.get(), p.1]);

            if let Ok([mut parent, mut sec]) = res {
                //parent.0.attacking = true;
                let dmg = (child.1.damage * time.delta_seconds()) * sec.1.protection;
                parent.2.filled += dmg/3.0;
                parent.1.health += dmg/2.0;
                sec.1.health -= dmg;
            }
        }

        else if q_detector.contains(p.0) && q_food.contains(p.1) {
            let child = q_detector.get_mut(p.0).unwrap();
            let res = q_minions.get_mut(child.0.get());
            let res_food = q_food.get_mut(p.1);

            if let (Ok(mut parent), Ok(mut food)) = (res, res_food) {
                //parent.0.eating = true;
                let dmg = (child.1.damage * time.delta_seconds()) * food.protection;
                parent.2.filled += dmg * 1.5;
                food.health -= dmg;
            }
        }

    });
}

fn detect( 
    rapier_context: Res<RapierContext>, 
    mut q_detector: Query<(&Parent, &Detector, &Collider)>,
    mut q_minions: Query<(&mut Brain, &Transform, &Sight), With<Minion>>,
    q_food: Query<&Transform, With<Food>>
){
    rapier_context.intersection_pairs().filter(|pp| pp.2 ).for_each(|pp|{
        let mut p = (pp.0, pp.1);

        if q_detector.contains(pp.1) {
            p = (pp.1, pp.0);
        }

        // Other minions
        if q_detector.contains(p.0) && q_minions.contains(p.1) {
            let child = q_detector.get_mut(p.0).unwrap();

            if child.0.get() != p.1 {
                let res = q_minions.get_many_mut([child.0.get(), p.1]);

                if let Ok([mut parent, sec]) = res {
                    let diff = sec.1.translation - parent.1.translation;
                    let dst = diff.length()/parent.2.radius/2.0;
                    let cos_theta = Vec2::new( parent.1.local_x().x, parent.1.local_x().y).normalize().dot((Vec2::new(diff.x, diff.y)).normalize());
                    let angle: f32 = cos_theta.acos() / PI - 0.5; // angle to minion

                    parent.0.sight_minions.push((angle, dst));
                }
            } 
        }

        // Food
        else if q_detector.contains(p.0) && q_food.contains(p.1) {
            let child = q_detector.get_mut(p.0).unwrap();
            let res = q_minions.get_mut(child.0.get());
            let res_food = q_food.get(p.1);

            if let (Ok(mut parent), Ok(food)) = (res, res_food) {

                let diff = food.translation - parent.1.translation;
                let dst = diff.length()/parent.2.radius/2.0;
                let cos_theta = Vec2::new( parent.1.local_x().x, parent.1.local_x().y).normalize().dot((Vec2::new(diff.x, diff.y)).normalize());
                let angle: f32 = cos_theta.acos() / PI - 0.5; // angle to minion

                parent.0.sight_food.push((angle, dst));
            }
        }

    });
}

fn reproduce(mut commands: Commands, 
    asset_server: Res<AssetServer>, 
    mut query: Query<(&Brain, &mut Hunger, &Transform), With<Minion>>
){
    query.iter_mut().for_each(|mut m|{
        if m.1.filled > 1.5 {
            m.1.filled -= 0.5;
            let pos = Vec2::new(m.2.translation.x, m.2.translation.y) + Vec2::new( m.2.local_x().x, m.2.local_x().y) * 500.0;
            spawn_minion(&mut commands, &asset_server, 
                &pos,
                m.0.clone()
            );
        }
    });
}

fn u_hunger(
    time: Res<Time>,
    mut query: Query<(&mut Hunger, &mut Health, &mut ExternalForce), With<Minion>>
) {
    query.par_for_each_mut(16, |(mut e, mut h, f)|{
        let energy = (f.force.y.abs() / 40.0) + (f.torque.abs() * 50.0);
        if e.filled <= 0.0 {
            h.health -= time.delta_seconds() * e.metabolism * energy + time.delta_seconds() / 20.0;
        } else {
            e.filled -= time.delta_seconds() * e.metabolism * energy + time.delta_seconds() / 20.0;
        }
        if e.filled > 2.1 {
            h.health += e.filled - 2.1;
            e.filled = 2.1;
        }
        if h.health > 1.0 {
            e.filled += h.health - 1.0;
            h.health = 1.0;
        }
    });
}

fn u_hp(
    mut commands: Commands,
    mut query: Query<(&mut Health, Entity)>
) {
    query.for_each_mut(|e|{
        if e.0.health <= 0.0 {commands.entity(e.1).despawn_recursive();}
    });
}

fn u_age(
    time: Res<Time>,
    mut query: Query<(&mut Age, &Brain), With<Minion>>,
) {
    query.par_for_each_mut(16, |(mut e, _)|{
        e.age += time.delta_seconds();
    });
}

fn u_sight(
    q_minions: Query<(&Children, &Sight), (With<Minion>, Changed<Sight>)>,
    mut q_detector: Query<&mut Collider, With<Detector>>,
){
    q_minions.iter().for_each(|c|{
        c.0.iter().for_each(|&id|{
            if q_detector.contains(id) {
                let mut child = q_detector.get_mut(id).unwrap();
                child.set_scale(Vec2::new(c.1.radius, c.1.radius), 1);
            }
        });
    });
}

fn u_weapon(
    mut q_weapons: Query<(&Weapon, &mut Collider), Changed<Weapon>>,
){
    q_weapons.par_for_each_mut(16, |mut w|{
        w.1.set_scale(Vec2::new(w.0.width, 4.0), 1);
    });
}

fn u_minion(
    mut q_minions: Query<(&Size, &mut Collider), Changed<Size>>,
){
    q_minions.par_for_each_mut(16, |mut m|{
        *m.1 = Collider::ball(m.0.radius);
        //m.1.set_scale(Vec2::new(64.0, 64.0), 1);
    });
}

fn u_food(time: Res<Time>,
    mut commands: Commands, 
    asset_server: Res<AssetServer>,
    query: Query<&Food>
) {
    let target_amount: i64;
    if time.elapsed_seconds() < 180.0 { target_amount = 512; }
    else if time.elapsed_seconds() < 360.0 { target_amount = 192; }
    else { target_amount = 64; }

    let amount_missing = target_amount - query.iter().len() as i64;

    if amount_missing > 0 { 
        //let norm = Normal::new(0.0, 0.8).unwrap();
        //let x = thread_rng().sample::<f32, _>(norm) - 0.5;
        //let y = thread_rng().sample::<f32, _>(norm) - 0.5;
        let position = Vec2::new((fr::f32()-0.5)*4000.0, (fr::f32() - 0.5)*4000.0);
        spawn_food(&mut commands, &asset_server, &position);
    }
}

fn save_best(
    mut best: ResMut<BestNN>,
    mut query: Query<(&Age, &mut Brain)>,
    keys: Res<Input<KeyCode>>,
){
    if keys.just_pressed(KeyCode::S) {
        let bb = query.iter().max_by(|a, b| a.0.age.partial_cmp(&b.0.age).unwrap()).unwrap();
        best.age = bb.0.age;
        best.nn = bb.1.nn.clone();
        best.nn.save("nn.dat");
        println!("Saved: {}", best.age);
    }
    if keys.just_pressed(KeyCode::L) {
        query.par_for_each_mut(2, |mut e|{
            e.1.nn.load("nn.dat");
        });
        println!("Loaded");
    }
}




pub struct MainPlugin;
impl Plugin for MainPlugin {
    fn build(&self, app: &mut App){
        app
            .add_startup_system(init_minions)
            //.add_startup_system(spawn_env)
            .add_system(u_food)
            .add_system(u_sight)
            .add_system(u_weapon)
            .add_system(u_minion)
            .add_system(u_age)
            .add_system(u_hunger)
            .add_system(u_hp)
            .add_system(ai)
            .add_system(detect)
            .add_system(attack)
            .add_system(reproduce)
            .add_system(movement)
            .add_system(save_best)
            .add_system(u_cursor)
            .insert_resource(BestNN::default())
        ;
    }
}

const BACKGROUND_COLOR: Color = Color::rgb(0.3, 0.3, 0.3);
fn main() {
    App::new()
        //.insert_resource(WindowDescriptor{scale_factor_override: Some(1.0),..default()})
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
        .add_plugin(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0))
        //.add_plugin(RapierDebugRenderPlugin::default())
        .insert_resource(RapierConfiguration{gravity: Vec2::new(0.0, 0.0), ..default()})
        .insert_resource(ClearColor(BACKGROUND_COLOR))
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(MainPlugin)
        .add_plugin(UiManPlugin)
        .add_system(bevy::window::close_on_esc)
        .run();
}

fn u_cursor(
    rapier_context: Res<RapierContext>,
    buttons: Res<Input<MouseButton>>,
    c_pos: Res<CursorWorld>,
    mut sel: ResMut<SelectedNN>,
    mut q_cursor: Query<&mut Transform, (Without<Minion>, With<Cursor>)>,
    q_minions: Query<(Entity, &Brain, &Transform, &Health, &Hunger, &Age), With<Minion>>
){
    if let Ok(mut cc) = q_cursor.get_single_mut() {
        let z = cc.translation.z;
        cc.translation = Vec3::new(c_pos.x, c_pos.y, z);

        if let Ok(minion) = q_minions.get(sel.eid) {
            sel.nn = minion.1.nn.clone();
            sel.pos = minion.2.translation.truncate();
            sel.hp = minion.3.health;
            sel.hunger = minion.4.filled;
            sel.age = minion.5.age;
        }
    }
    
    if buttons.just_pressed(MouseButton::Left) {
        rapier_context.intersection_pairs().filter(|pp| pp.2 ).for_each(|pp|{
            let mut p = (pp.0, pp.1);

            if q_cursor.contains(pp.1) {
                p = (pp.1, pp.0);
            }

            if q_cursor.contains(p.0) && q_minions.contains(p.1) {
                let minion = q_minions.get(p.1).unwrap();
                sel.eid = minion.0;
            }

        });
    }
}


/*  GROUPS   
*  0 - everyone
*  1 - enviroment
*  2 - minions 
*  3 - food(A)
*  4 - food(B)
*  */



fn spawn_food(commands: &mut Commands, asset_server: &Res<AssetServer>, position: &Vec2) {
    let member = Group::from_bits(0b10010000000000000000000000000000).unwrap();
    let filter = Group::from_bits(0b11111000000000000000000000000000).unwrap();
    commands.spawn(Food) 
        .insert(Health::new(0.99))
        .insert(Size{radius: 8.0})
        .insert(RigidBody::Dynamic)
        .insert(Damping {linear_damping: 0.98, angular_damping: 0.98 })
        .insert(Collider::ball(8.0))
        .insert(CollisionGroups::new(member, filter))
        .insert(Restitution::coefficient(0.0))
        .insert(Friction::coefficient(0.5))
        .insert(SpriteBundle {
            transform: Transform::from_translation(Vec3::new(position.x, position.y, 0.0)),
            texture: asset_server.load("../assets/textures/Food_a.png"),
            ..default()
        })
        ;
}


fn init_minions(mut commands: Commands, asset_server: Res<AssetServer>) {
    for i in 0..420 {
        //let norm = Normal::new(0.0, 0.8).unwrap();
        //let x = thread_rng().sample::<f32, _>(norm) - 0.5;
        //let y = thread_rng().sample::<f32, _>(norm) - 0.5;
        let position = Vec2::new((fr::f32() - 0.5)*4000.0, (fr::f32() - 0.5)*4000.0);

        let p = spawn_minion(&mut commands, &asset_server, &position, Brain::default());
        if i == -1 {commands.entity(p).insert(Control);}
    }
    let member = Group::from_bits(0b00100000000000000000000000000000).unwrap();
    let filter = Group::from_bits(0b00100000000000000000000000000000).unwrap();
    commands.spawn(Cursor)
        .insert(Collider::ball(1.0))
        .insert(Sensor)
        .insert(CollisionGroups::new(member, filter))
        .insert(SpriteBundle {
            transform: Transform::default(),
            texture: asset_server.load("../assets/textures/Cursor.png"),
            ..default()})
    ;
}

// It is NOT a system, but to be used by systems to spawn minions
fn spawn_minion(commands: &mut Commands, asset_server: &Res<AssetServer>, position: &Vec2, brain: Brain) -> Entity {
    let mut brain = brain;
    brain.nn.mutate();
    brain.nn.mutate();
    brain.nn.mutate();
    let member = Group::from_bits(0b10100000000000000000000000000000).unwrap();
    let filter = Group::from_bits(0b11111000000000000000000000000000).unwrap();
    let p = commands.spawn(Minion) 
        .insert(brain)
        .insert(Health::default())
        .insert(Age::default())
        .insert(Hunger::default())
        .insert(Sight::default())
        .insert(Size::default())
        .insert(RigidBody::Dynamic)
        .insert(Velocity::default())
        .insert(ExternalForce::default())
        .insert(Damping {linear_damping: 0.98, angular_damping: 0.98 })
        .insert(Collider::ball(16.0))
        .insert(CollisionGroups::new(member, filter))
        .insert(Restitution::coefficient(0.0))
        .insert(Friction::coefficient(0.5))
        .insert(SpriteBundle {
            transform: Transform::from_translation(Vec3::new(position.x, position.y, 0.0)),
            texture: asset_server.load("../assets/textures/Minion.png"),
            ..default()
        })
        .id();
        
    let joint_d = FixedJointBuilder::new().local_anchor1(Vec2::new(0.0, 30.0));
    commands.spawn(Detector)
        .insert(RigidBody::Dynamic)
        .insert(Collider::ball(1.0))
        .insert(ColliderMassProperties::Mass(0.000001))
        .insert(Sensor)
        .set_parent(p)
        .insert(ImpulseJoint::new(p, joint_d));

    let joint_w = FixedJointBuilder::new().local_anchor1(Vec2::new(0.0, 24.0));
    commands.spawn(Weapon::default())
        .insert(RigidBody::Dynamic)
        .insert(Collider::cuboid(1.0, 1.0))
        .insert(ColliderMassProperties::Mass(0.000001))
        .insert(Sensor)
        .set_parent(p)
        .insert(ImpulseJoint::new(p, joint_w));
    p
}

fn movement( 
    mut query: Query<(&mut Brain, &mut ExternalForce, &Transform), With<Control>>,
    keyboard_input: Res<Input<KeyCode>>
) {
    query.for_each_mut(|(mut brain, mut force, transform)| {
        brain.nn.forward(&[1.0, 0.4]);

        force.force = Vec2::default();
        force.torque = 0.0;
        if keyboard_input.pressed(KeyCode::Left){ force.torque = 0.002;}
        if keyboard_input.pressed(KeyCode::Right){ force.torque = -0.002; }
        if keyboard_input.pressed(KeyCode::Up){ force.force.y = 20.0; }
        if keyboard_input.pressed(KeyCode::Down){ force.force.y = -20.0; }
        let dir = Vec2::new( transform.local_x().x, transform.local_x().y);
        force.force = force.force.rotate(dir);
    });
}

//fn spawn_env(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<ColorMaterial>>) {
//    let member = Group::from_bits(0b11000000000000000000000000000000).unwrap();
//    let filter = Group::from_bits(0b11111000000000000000000000000000).unwrap();
//    commands
//        .spawn(Collider::cuboid(500.0, 50.0))
//        .insert(CollisionGroups::new(member, filter))
//        .insert(MaterialMesh2dBundle {
//            mesh: meshes.add(shape::Quad::new(Vec2::new(500.0, 50.0)*2.0).into()).into(),
//            material: materials.add(ColorMaterial::from(Color::NAVY)),
//            transform: Transform::from_translation(Vec3::new(800., 0., 0.)),
//            ..default()});
//    commands
//        .spawn(Collider::cuboid(100.0, 100.0))
//        .insert(CollisionGroups::new(member, filter))
//        .insert(MaterialMesh2dBundle {
//            mesh: meshes.add(shape::Quad::new(Vec2::new(100.0, 100.0)*2.0).into()).into(),
//            material: materials.add(ColorMaterial::from(Color::NAVY)),
//            transform: Transform::from_translation(Vec3::new(900., 600., 0.)),
//            ..default()});
//}






//fn collisions(mut commands: Commands, 
//    time: Res<Time>,
//    rapier_context: Res<RapierContext>, 
//    mut query: Query<(&mut Health, &Weapon), With<Minion>>
//){
//    rapier_context.contact_pairs().for_each(|p|{
//        if query.contains(p.collider1()) && query.contains(p.collider2()) {
//
//            let [mut m1, mut m2] = query.get_many_mut([p.collider1(), p.collider2()]).unwrap();
//            m1.0.health -= (m2.1.damage * time.delta_seconds())*m1.0.protection;
//            m2.0.health -= (m1.1.damage * time.delta_seconds())*m1.0.protection;
//            if m1.0.health <= 0.0 {commands.entity(p.collider1()).despawn_recursive();}
//            if m2.0.health <= 0.0 {commands.entity(p.collider2()).despawn_recursive();}
//        }
//    });
//}

//fn cast_ray(
//    rapier_context: Res<RapierContext>,
//    mut query: Query<(&mut Brain, &Transform, Entity), With<Minion>>
//) {
//    query.par_for_each_mut(1, |(mut brain, transform, e)| {
//        let ray_pos = Vec2::new( transform.translation.x, transform.translation.y );
//        let ray_dir = Vec2::new( transform.local_x().x, transform.local_x().y );
//        let max_toi = brain.dof;
//        let solid = true;
//        brain.sight.par_iter_mut().for_each(|s|{
//            s.0 = RayType::None;
//            s.1 = f32::MAX;
//        });
//        brain.sight.iter_mut().enumerate().for_each(|(i, s)|{
//            let member = Group::from_bits(0b00100000000000000000000000000000).unwrap();
//            let filter = Group::from_bits(0b00100000000000000000000000000000).unwrap();
//            let ff = QueryFilter::default().exclude_collider(e).groups(CollisionGroups::new(member, filter));
//            
//            if let Some((_entity, toi)) = rapier_context.cast_ray(ray_pos, ray_dir, max_toi, solid, ff) {
//                //let hit_point = ray_pos + ray_dir * toi;
//                s.0 = RayType::Minion;
//                s.1 = toi;
//            } 
//        });
//
//    });
//}
