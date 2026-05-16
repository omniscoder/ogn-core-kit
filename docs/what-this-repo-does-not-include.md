# What this repo does not include

OGN Core Kit is the public adoption layer. It intentionally does not ship:

- the proprietary OGN Engine
- CUDA kernels
- commercial engine containers
- hosted OGN Cloud control plane
- private billing/auth policy implementation

The private engine repository remains the production implementation
that maps to the same Job Spec JSON v1 contract.
