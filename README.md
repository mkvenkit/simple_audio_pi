## Inference

Expected Input:

```
[
    {
        'name': 'input_3', 
        'index': 0, 
        'shape': array([  1, 129, 124,   1]), 
        'shape_signature': array([  1, 129, 124,   1]), 
        'dtype': <class 'numpy.float32'>, 
        'quantization': (0.0, 0), 
        'quantization_parameters': {'scales': array([], dtype=float32), 
        'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 
        'sparsity_parameters': {}
    }
]
```

Expected Output

```
[
    {
        'name': 'Identity', 
        'index': 17, 
        'shape': array([1, 8]), 
        'shape_signature': array([1, 8]), 
        'dtype': <class 'numpy.float32'>, 
        'quantization': (0.0, 0), 
        'quantization_parameters': {'scales': array([], dtype=float32), 
        'zero_points': array([], dtype=int32), 
        'quantized_dimension': 0}, 
        'sparsity_parameters': {}
    }
]
```