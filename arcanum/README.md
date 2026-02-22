# Arcanum

Formal verification tool for a safe C++ subset targeting safety-critical automotive software (ISO 26262).

## Prerequisites

Build inside the Docker environment:

```bash
cd arcanum/docker
./docker_build.sh
./docker_run.sh
docker exec -it arcanum-dev bash
```

## Build

```bash
cd /workspace/ad-adas-memo/arcanum
cmake --preset default
cmake --build build/default
```

## Test

```bash
ctest --preset default
```

## Usage (Slice 1)

```bash
./build/default/bin/arcanum --mode=verify input.cpp
```
