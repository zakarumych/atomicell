//! This crate provides [`AtomicCell`] type - a multi-threaded version of [`RefCell`] from standard library.
//!
//! [`AtomicCell`] uses atomics to track borrows and is able to guarantee
//! absence of mutable aliasing when multiple threads try to borrow concurrently.
//!
//! Additionally it exports borrowing primitives [`AtomicBorrow`] and [`AtomicBorrowMut`] in [`borrow`] module.
//! These types can be used to build other abstractions with atomic borrowing capabilities.
//!
//! [`RefCell`]: https://doc.rust-lang.org/core/cell/struct.RefCell.html
//! [`AtomicBorrow`]: borrow/struct.AtomicBorrow.html
//! [`AtomicBorrowMut`]: borrow/struct.AtomicBorrowMut.html

#![no_std]

pub mod borrow;
mod cell;
mod refs;
#[cfg(test)]
mod tests;

pub use self::{
    cell::AtomicCell,
    refs::{Ref, RefMut},
};
