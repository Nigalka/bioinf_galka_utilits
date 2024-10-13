from my_modules import dna_rna_tools
from my_modules import mod_filter_fastq


def run_dna_rna_tools(*dna_tools: tuple) -> str | list:
    """ Hello! Function run_dna_rna_tools works with dna and rna sequences
    And have subfunctions reverse_complement, complement, reverse, transcribe


    Args: *dna_tools: str


    Returns: str or list
    """
    *sequences, operation = dna_rna_tools.proverka(dna_tools)
    if operation == 'reverse_complement':
        ret = dna_rna_tools.reverse_complement(sequences)
    if operation == 'complement':
        ret = dna_rna_tools.complement(sequences)
    if operation == 'reverse':
        ret = dna_rna_tools.reverse(sequences)
    if operation == 'transcribe':
        ret = dna_rna_tools.transcribe(sequences)
    if len(ret) == 1:
        return ret[0]
    return ret


def filter_fastq(seqs: dict, gc_bounds: int | tuple = (0, 100),
                 length_bounds: int | tuple = (0, 2**32),
                 quality_threshold: int | tuple = 0) -> dict:
    """Function filter_fastq returns dictionary with sequences
    that fulfill conditions

    Args: seqs:dict
          gc_bounds:int | tuple = (0, 100)
          length_bounds:int | tuple = (0, 2**32)
          quality_threshold:int | tuple = 0

        Returns: dict
    """
    seqs_fun = mod_filter_fastq.seqs_right_quality(seqs, quality_threshold)
    seqs_fun = mod_filter_fastq.seqs_right_gc_bounds(seqs_fun, gc_bounds)
    seqs_fun = mod_filter_fastq.seqs_right_length(seqs_fun, length_bounds)
    return seqs_fun
