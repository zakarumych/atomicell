# Atomicell crate

[![crates](https://img.shields.io/crates/v/atomicell.svg?style=for-the-badge&label=atomicell)](https://crates.io/crates/atomicell)
[![docs](https://img.shields.io/badge/docs.rs-atomicell-66c2a5?style=for-the-badge&labelColor=555555&logoColor=white)](https://docs.rs/atomicell)
[![actions](https://img.shields.io/github/actions/workflow/status/zakarumych/atomicell/badge.yml?branch=master&style=for-the-badge)](https://github.com/zakarumych/atomicell/actions/workflows/badge.yml)
[![MIT/Apache](https://img.shields.io/badge/license-MIT%2FApache-blue.svg?style=for-the-badge)](COPYING)
![loc](https://img.shields.io/tokei/lines/github/zakarumych/atomicell?style=for-the-badge)

This crate provides `AtomicCell` type - a multi-threaded version of `RefCell` from standard library.
`AtomicCell` uses atomics to track borrows and able to guarantee
absence of mutable aliasing when multiple threads try to borrow concurrently.

Unlike mutexes and spin-locks `AtomicCell` does not have blocking calls.
Borrows are either succeed immediately or fail.

There are fallible that return optional for borrowing calls - [`AtomicCell::try_borrow`] and [`AtomicCell::try_borrow_mut`].

And panicking version - [`AtomicCell::borrow`] and [`AtomicCell::borrow_mut`].

[`AtomicCell::try_borrow`]: https://docs.rs/atomicell/latest/atomicell/struct.AtomicCell.html#method.try_borrow
[`AtomicCell::try_borrow_mut`]: https://docs.rs/atomicell/latest/atomicell/struct.AtomicCell.html#method.try_borrow_mut
[`AtomicCell::borrow`]: https://docs.rs/atomicell/latest/atomicell/struct.AtomicCell.html#method.borrow
[`AtomicCell::borrow_mut`]: https://docs.rs/atomicell/latest/atomicell/struct.AtomicCell.html#method.borrow_mut
