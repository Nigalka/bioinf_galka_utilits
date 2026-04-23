from __future__ import annotations
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import ClassVar, Dict, Iterator, Mapping, Union
from pathlib import Path
import gzip
from statistics import mean
import logging
import argparse
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import gc_fraction

SliceOrInt = Union[slice, int]

# Настройка логирования в файл
logging.basicConfig(
    filename="fastq_filter.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

@dataclass(frozen=True, slots=True)
class BiologicalSequence(ABC):
    """Abstract class for biological sequences."""
    _content: str

    def __post_init__(self) -> None:
        if not isinstance(self._content, str):
            raise TypeError("Sequence data must be a string")
        if len(self._content) == 0:
            raise ValueError("Sequence cannot be empty")
        if not self.check_alphabet():
            raise ValueError(f"Invalid characters for {self.__class__.__name__}: {self._content!r}")

    def __len__(self) -> int:
        return len(self._content)

    def __getitem__(self, idx: SliceOrInt) -> Union[str, BiologicalSequence]:
        if isinstance(idx, slice):
            return self.__class__(self._content[idx])
        return self._content[idx]

    def __iter__(self) -> Iterator[str]:
        return iter(self._content)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}('{self._content}')"

    def __repr__(self) -> str:
        return str(self)

    @abstractmethod
    def check_alphabet(self) -> bool:
        raise NotImplementedError


class NucleicAcidSequence(BiologicalSequence, ABC):
    _allowed_chars: ClassVar[frozenset[str]]
    _base_pairs: ClassVar[Mapping[str, str]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_content", self._content.upper())
        super().__post_init__()

    def check_alphabet(self) -> bool:
        if self.__class__ is NucleicAcidSequence:
            raise NotImplementedError("NucleicAcidSequence is abstract; use DNASequence or RNASequence")
        return set(self._content).issubset(self._allowed_chars)

    def complement(self) -> NucleicAcidSequence:
        if self.__class__ is NucleicAcidSequence:
            raise NotImplementedError("NucleicAcidSequence is abstract; use DNASequence or RNASequence")
        paired = "".join(self._base_pairs[base] for base in self._content)
        return self.__class__(paired)

    def reverse(self) -> NucleicAcidSequence:
        if self.__class__ is NucleicAcidSequence:
            raise NotImplementedError("NucleicAcidSequence is abstract; use DNASequence or RNASequence")
        return self.__class__(self._content[::-1])

    def reverse_complement(self) -> NucleicAcidSequence:
        return self.reverse().complement()


class DNASequence(NucleicAcidSequence):
    _allowed_chars: ClassVar[frozenset[str]] = frozenset({"A", "T", "G", "C"})
    _base_pairs: ClassVar[Mapping[str, str]] = {"A": "T", "T": "A", "G": "C", "C": "G"}

    def transcribe(self) -> RNASequence:
        return RNASequence(self._content.replace("T", "U"))


class RNASequence(NucleicAcidSequence):
    _allowed_chars: ClassVar[frozenset[str]] = frozenset({"A", "U", "G", "C"})
    _base_pairs: ClassVar[Mapping[str, str]] = {"A": "U", "U": "A", "G": "C", "C": "G"}


class AminoAcidSequence(BiologicalSequence):
    _allowed_chars: ClassVar[frozenset[str]] = frozenset(
        {"A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
         "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_content", self._content.upper())
        super().__post_init__()

    def check_alphabet(self) -> bool:
        return set(self._content).issubset(self._allowed_chars)

    def get_aa_composition(self) -> Dict[str, int]:
        return dict(Counter(self._content))


def _open_file_with_gzip_support(filepath: Path, mode: str):
    if filepath.suffix == ".gz":
        return gzip.open(filepath, mode)
    return filepath.open(mode)


def _parse_range_constraints(constraints, low_default: float, high_default: float) -> tuple[float, float]:
    if constraints is None:
        return low_default, high_default
    if isinstance(constraints, (int, float)):
        return low_default, float(constraints)
    if isinstance(constraints, (tuple, list)) and len(constraints) == 2:
        return float(constraints[0]), float(constraints[1])
    raise ValueError("Constraints must be a number, 2-tuple/list, or None")


def filter_fastq(
    source_file: str,
    destination_file: str,
    gc_constraints=(0.0, 1.0),
    length_constraints=(0, 2**32),
    quality_cutoff: float = 0.0,
) -> dict:
    
    src_path = Path(source_file)
    dst_path_raw = Path(destination_file)

    if not src_path.exists():
        logging.error(f"Input file not found: {src_path}")
        raise FileNotFoundError(f"Source file does not exist: {src_path}")

    output_directory = dst_path_raw.parent / "filtered"
    output_directory.mkdir(parents=True, exist_ok=True)
    final_path = output_directory / dst_path_raw.name

    gc_min, gc_max = _parse_range_constraints(gc_constraints, 0.0, 1.0)
    len_min, len_max = _parse_range_constraints(length_constraints, 0.0, float(2**32))

    total_count = 0
    passed_count = 0

    def meets_criteria(record: SeqRecord) -> bool:
        read_length = len(record.seq)
        if not (len_min <= read_length <= len_max):
            return False

        gc_value = gc_fraction(record.seq)
        if not (gc_min <= gc_value <= gc_max):
            return False

        quality_values = record.letter_annotations.get("phred_quality")
        if not quality_values:
            return False
        if mean(quality_values) < quality_cutoff:
            return False

        return True

    logging.info(f"Starting FASTQ filtering for {src_path}")
    
    with _open_file_with_gzip_support(src_path, "rt") as input_handle, \
         _open_file_with_gzip_support(final_path, "wt") as output_handle:
        
        record_iterator = SeqIO.parse(input_handle, "fastq")
        write_records = SeqIO.write

        batch: list[SeqRecord] = []
        for record in record_iterator:
            total_count += 1
            if meets_criteria(record):
                passed_count += 1
                batch.append(record)

                if len(batch) >= 1000:
                    write_records(batch, output_handle, "fastq")
                    batch.clear()

        if batch:
            write_records(batch, output_handle, "fastq")

    stats = {
        "input_fastq": str(src_path),
        "output_fastq": str(final_path),
        "total_reads": total_count,
        "passed_reads": passed_count,
        "rejected_reads": total_count - passed_count,
        "pass_ratio": (passed_count / total_count) if total_count else 0.0,
        "gc_range": (gc_min, gc_max),
        "length_range": (int(len_min), int(len_max)),
        "quality_cutoff": quality_cutoff,
    }
    
    logging.info(f"Filtering complete. Passed: {passed_count}/{total_count}")
    return stats

def main():
    parser = argparse.ArgumentParser(description="Filter FASTQ files based on GC content, length, and quality.")
    parser.add_argument("-i", "--input", required=True, help="Path to input FASTQ file")
    parser.add_argument("-o", "--output", required=True, help="Name of output FASTQ file")
    parser.add_argument("--gc", nargs=2, type=float, help="GC constraints (min max). Example: --gc 0.4 0.6", default=(0.0, 1.0))
    parser.add_argument("--len", nargs=2, type=int, help="Length constraints (min max).", default=(0, 2**32))
    parser.add_argument("--qual", type=float, help="Mean Phred quality cutoff.", default=0.0)

    args = parser.parse_args()

    try:
        results = filter_fastq(
            source_file=args.input,
            destination_file=args.output,
            gc_constraints=tuple(args.gc),
            length_constraints=tuple(args.len),
            quality_cutoff=args.qual
        )
        print("Filtering successful! Stats:")
        for key, val in results.items():
            print(f"{key}: {val}")
    except Exception as e:
        logging.error(f"Execution failed: {e}", exc_info=True)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()