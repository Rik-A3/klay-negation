from jax.ops import segment_min, segment_max

def min_layer(num_segments, ix_in, ix_out, x):
    return segment_min(x[ix_in], ix_out, num_segments=num_segments, indices_are_sorted=True)


def max_layer(num_segments, ix_in, ix_out, x):
    return segment_max(x[ix_in], ix_out, num_segments=num_segments, indices_are_sorted=True)
