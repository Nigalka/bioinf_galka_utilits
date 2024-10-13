def seqs_right_quality(seqs: dict, quality_threshold: int) -> dict:
    medium_quality = 0
    seqs_to_ret = {}
    for sequance in seqs:
        for i in range(len(seqs[sequance][1])):
            medium_quality = medium_quality + ord(seqs[sequance][1][i]) - 33
        medium_quality = medium_quality/(i+1)
        if quality_threshold < medium_quality:
            seqs_to_ret[sequance] = seqs[sequance]
    return seqs_to_ret


def seqs_right_gc_bounds(seqs: dict, gc_bounds: int | tuple) -> dict:
    seqs_to_ret = {}
    for sequance in seqs:
        g_bounds_per_seq = seqs[sequance][0].count('G')
        gc_bounds_per_seq = g_bounds_per_seq + seqs[sequance][0].count('C')
        gc_bounds_per_seq = 100 * gc_bounds_per_seq/len(seqs[sequance][0])
        if gc_bounds is not int:
            if gc_bounds[0] <= gc_bounds_per_seq <= gc_bounds[1]:
                seqs_to_ret[sequance] = seqs[sequance]
        else:
            if gc_bounds_per_seq <= gc_bounds:
                seqs_to_ret[sequance] = seqs[sequance]
    return seqs_to_ret


def seqs_right_length(seqs: dict,
                      length_bounds: int | tuple) -> dict:
    seqs_to_ret = {}
    for sequance in seqs:
        len_seq = len(seqs[sequance][0])
        if length_bounds is not int:
            if length_bounds[0] <= len_seq <= length_bounds[1]:
                seqs_to_ret[sequance] = seqs[sequance]
        else:
            if len_seq <= length_bounds:
                seqs_to_ret[sequance] = seqs[sequance]
    return seqs_to_ret
