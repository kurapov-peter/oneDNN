A copy of the ['graph compiler' backend](https://github.com/dchigarev/oneDNN/tree/init_elyzor/src/graph/backend/graph_compiler) without an actual compiler.

#### How to enable:
Pass `-DONEDNN_EXPERIMENTAL_ELYZOR_BACKEND=ON` to your cmake:
```
cd oneDNN
mkdir build && cd build
cmake ../ -DONEDNN_EXPERIMENTAL_ELYZOR_BACKEND=ON
```

The current implementation of `elyzor/CMakeLists.txt` looks for [graph-compiler](https://github.com/intel/graph-compiler/tree/AndreyPavlenko-dev)
at the following paths:
```
oneDNN/../graph-compiler/include/graph_compiler.h
oneDNN/../graph-compiler/build/libgraph_compiler.so
```
Make sure they're available.

#### How to test:
There's an example file that uses elyzor backend for compilation/execution ([examples/graph/cpu_elyzor_test.cpp](https://github.com/dchigarev/oneDNN/blob/init_elyzor/examples/graph/cpu_elyzor_test.cpp)).

Currently, it's only able to print "hello world" strings from [compile](https://github.com/dchigarev/oneDNN/blob/c0a48558295dfcabf84c6ab68e6311ac95c98d6b/src/graph/backend/elyzor/compiler_partition_impl.cpp#L121) and [execute](https://github.com/dchigarev/oneDNN/blob/c0a48558295dfcabf84c6ab68e6311ac95c98d6b/src/graph/backend/elyzor/compiler_partition_impl.cpp#L185) methods.

#### Hacks:
1. The graph compiler's front-end [uses certain functionality](https://github.com/dchigarev/oneDNN/blob/c0a48558295dfcabf84c6ab68e6311ac95c98d6b/src/graph/backend/graph_compiler/target_machine.hpp#L19-L24)
   from its core to detect which CPU instructions are available and [define patterns accordingly](https://github.com/dchigarev/oneDNN/blob/c0a48558295dfcabf84c6ab68e6311ac95c98d6b/src/graph/backend/graph_compiler/compiler_backend.cpp#L54).
   In elyzor we don't have this functionality, so we are [assuming that all instructions are available](https://github.com/dchigarev/oneDNN/blob/c0a48558295dfcabf84c6ab68e6311ac95c98d6b/src/graph/backend/elyzor/target_machine.hpp#L19-L27).
