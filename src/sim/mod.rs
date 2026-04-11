pub mod mutation;
pub mod rule;
pub mod world;

pub use mutation::{MutationQueue, WorldMutation};
pub use rule::{CaRule, GameOfLife3D};
pub use world::World;
