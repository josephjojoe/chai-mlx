def forward_256(self,
    input: Tensor) -> Tensor:
  weight = self.weight
  return torch.linear(input, weight)
