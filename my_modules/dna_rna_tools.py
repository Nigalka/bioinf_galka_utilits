dna_dna_complementary = {'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'A': 'T',
                         'T': 'A', 'C': 'G', 'G': 'C'}
rna_rna_complementary = {'a': 'u', 'u': 'a', 'c': 'g', 'g': 'c', 'A': 'U',
                         'U': 'A', 'C': 'G', 'G': 'C'}


def transcribe(dna: tuple) -> list:
    dna_rna_transcribe = {'t': 'u', 'T': 'U'}
    rna_dna_transcribe = {'u': 't', 'U': 'T'}
    rna = []
    for nuc_ac in dna:
        has_t = 'T' in nuc_ac.upper()
        has_u = 'U' in nuc_ac.upper()
        if has_t:
            nuc_ac = nuc_ac.translate(str.maketrans(dna_rna_transcribe))
        elif has_u:
            nuc_ac = nuc_ac.translate(str.maketrans(rna_dna_transcribe))
        rna.append(nuc_ac)
    return (rna)


def reverse(dna: tuple) -> list:
    rna = []
    for nuc_ac in dna:
        nuc_ac = nuc_ac[::-1]
        rna.append(nuc_ac)
    return (rna)


def complement(dna: tuple) -> list:
    rna = []
    for nuc_ac in dna:
        has_t = 'T' in nuc_ac.upper()
        has_u = 'U' in nuc_ac.upper()
        if has_t:
            nuc_ac = nuc_ac.translate(str.maketrans(dna_dna_complementary))
        elif has_u:
            nuc_ac = nuc_ac.translate(str.maketrans(rna_rna_complementary))
        rna.append(nuc_ac)
    return (rna)


def reverse_complement(dna: tuple) -> list:
    rna = []
    for nuc_ac in dna:
        if nuc_ac.find('t') != -1 or nuc_ac.find('T') != -1:
            nuc_ac = nuc_ac.translate(str.maketrans(dna_dna_complementary))
        elif nuc_ac.find('u') != -1 or nuc_ac.find('U') != -1:
            nuc_ac = nuc_ac.translate(str.maketrans(rna_rna_complementary))
        nuc_ac = nuc_ac[::-1]
    return (rna)


def proverka(dna_tools: tuple) -> list:
    dna = []
    valid_symbols = 'acgtuACGTU'
    for nuc_ac in dna_tools[:-1]:
        if any(ch not in valid_symbols for ch in nuc_ac):
            continue  # found invalid symbol

        has_t = 'T' in nuc_ac.upper()
        has_u = 'U' in nuc_ac.upper()
        if has_t and has_u:
            continue  # contains both T and U, which is not possible
    # everything is ok, so append this sequence
        dna.append(nuc_ac)
    dna.append(dna_tools[-1])
    return (dna)
