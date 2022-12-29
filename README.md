# Atomicell crate

[![crates](https://img.shields.io/crates/v/atomicell.svg?style=for-the-badge&label=atomicell)](https://crates.io/crates/atomicell)
[![docs](https://img.shields.io/badge/docs.rs-atomicell-66c2a5?style=for-the-badge&labelColor=555555&logoColor=white)](https://docs.rs/atomicell)
[![actions](https://img.shields.io/github/workflow/status/zakarumych/atomicell/badge/master?style=for-the-badge)](https://github.com/zakarumych/atomicell/actions?query=workflow%3ARust)
[![MIT/Apache](https://img.shields.io/badge/license-MIT%2FApache-blue.svg?style=for-the-badge)](COPYING)
![loc](https://img.shields.io/tokei/lines/github/zakarumych/atomicell?style=for-the-badge)

This crate provides `AtomicCell` type - a multi-threaded version of `RefCell` from standard library.
`AtomicCell` uses atomics to track borrows and thus able guarantee
absence of mutable aliasing when multiple threads try to borrow concurrently.
