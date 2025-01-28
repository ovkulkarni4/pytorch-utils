from dataclasses import asdict
from typing import List, Optional, Union, Callable
from enum import Enum
import torch
import torch.nn as nn
from functools import reduce

class LayerType(Enum):
    """Enumeration of supported neural network layer types."""
    LINEAR = "linear"

@dataclasses.dataclass
class LayerSpecs:
    """Specifications for a neural network layer.

    Attributes:
        in_features (int): Number of input features for the layer
        out_features (int): Number of output features for the layer
    """
    in_features: int
    out_features: int

class GenericNeuralNetworkModel(nn.Module):
    """A generic neural network model that can be configured with various architectures.
    
    This class provides a flexible way to create neural networks with different layer 
    configurations and activation functions. It supports various layer types and can be
    easily extended to support more complex architectures.

    Attributes:
        layers (nn.ModuleList): List of neural network layers
        activation_fn (Optional[nn.Module]): Activation function to be applied between layers
        
    Args:
        specs (List[LayerSpecs]): List of layer specifications defining the network architecture.
            Each spec should contain the input and output dimensions for each layer.
        activation_fn (Optional[nn.Module]): Activation function to use between layers.
            If None, no activation function is applied. Default: None
        layer_type (LayerType): Type of layers to use in the network.
            Currently supports LINEAR layers only. Default: LayerType.LINEAR

    Example:
        >>> model = GenericNeuralNetworkModel(
        ...     specs=[
        ...         LayerSpecs(2, 10),  # Input layer: 2 features -> 10 features
        ...         LayerSpecs(10, 10), # Hidden layer: 10 features -> 10 features
        ...         LayerSpecs(10, 1)   # Output layer: 10 features -> 1 feature
        ...     ],
        ...     activation_fn=nn.ReLU()
        ... )
    
    Raises:
        ValueError: If an unsupported layer type is specified
    """
    
    def __init__(
        self, 
        specs: List[LayerSpecs], 
        activation_fn: Optional[nn.Module] = None,
        layer_type: LayerType = LayerType.LINEAR
    ):
        super().__init__()
        self.layers = self._generate_layers(specs, layer_type)
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Applies each layer sequentially, with activation functions between layers
        except for the final layer.
        
        Args:
            x (torch.Tensor): Input tensor with shape matching the first layer's
                input dimensions

        Returns:
            torch.Tensor: Output tensor with shape matching the final layer's
                output dimensions
        """
        return compose(
            *([zip(layer, self.activation_fn) for layer in self.layers][:-1])
        )

    def _generate_layers(
        self, 
        specs: List[LayerSpecs], 
        layer_type: LayerType
    ) -> nn.ModuleList:
        """Generates neural network layers based on specifications.
        
        Args:
            specs (List[LayerSpecs]): List of layer specifications
            layer_type (LayerType): Type of layers to generate

        Returns:
            nn.ModuleList: List of generated neural network layers

        Raises:
            ValueError: If layer_type is not supported
        """
        layers = []
        for i, spec in enumerate(specs):
            layer = None

            if layer_type == LayerType.LINEAR:
                layer = nn.Linear(**asdict(spec))
            else:
                raise ValueError(f"Unknown or unsupported layer type: {layer_type}")

            setattr(self, f"layer_{i}", layer)
            layers.append(layer)
            
        return nn.ModuleList(layers)

def compose(*functions: List[Callable]) -> Callable:
    """Composes multiple functions into a single function.
    
    Args:
        *functions: Variable number of function pairs (layer, activation)
        
    Returns:
        Callable: Composed function that applies all functions in sequence
    """
    return reduce(lambda f, g: lambda x: g(f(x)), functions)

# Example usage
if __name__ == "__main__":
    model = GenericNeuralNetworkModel(
        specs=[
            LayerSpecs(2, 10),
            LayerSpecs(10, 10),
            LayerSpecs(10, 1)
        ],
        activation_fn=nn.ReLU()
    )
