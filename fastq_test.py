import pytest
from pathlib import Path
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

# Импортируем из нашего основного файла (предполагается, что он называется main.py)
from main import (
    DNASequence,
    RNASequence,
    AminoAcidSequence,
    _parse_range_constraints,
    filter_fastq
)

class TestBiologicalSequences:
    """Тесты для проверки логики классов биологических последовательностей"""
    
    def test_empty_sequence_error(self):
        """1. Проверка создания пустой последовательности (должно быть исключение)"""
        with pytest.raises(ValueError, match="Sequence cannot be empty"):
            DNASequence("")

    def test_invalid_alphabet_error(self):
        """2. Проверка невалидных нуклеотидов"""
        with pytest.raises(ValueError, match="Invalid characters"):
            DNASequence("ATGCZ")

    def test_dna_transcribe(self):
        """3. Транскрипция DNA в RNA"""
        dna = DNASequence("ATGC")
        rna = dna.transcribe()
        assert isinstance(rna, RNASequence)
        assert str(rna) == "RNASequence('AUGC')"

    def test_amino_acid_composition(self):
        """4. Подсчет аминокислот"""
        aa = AminoAcidSequence("AARRGG")
        comp = aa.get_aa_composition()
        assert comp == {"A": 2, "R": 2, "G": 2}

class TestFastqFilter:
    """Тесты для тула фильтрации FASTQ и его вспомогательных функций"""

    def test_parse_range_constraints_invalid(self):
        """5. Передача неверного формата ограничений"""
        with pytest.raises(ValueError, match="Constraints must be a number, 2-tuple/list, or None"):
            _parse_range_constraints("строка", 0.0, 1.0)

    @pytest.fixture
    def sample_fastq_file(self, tmp_path):
        """структура для создания временного тестового FASTQ файла"""
        file_path = tmp_path / "test_input.fastq"
        
        records = [
            # Хороший рид 
            SeqRecord(Seq("ATGCATGCAT"), id="seq1", letter_annotations={"phred_quality": [30]*10}),
            # Плохое качество
            SeqRecord(Seq("ATGCATGCAT"), id="seq2", letter_annotations={"phred_quality": [10]*10}),
            # Слишком короткий
            SeqRecord(Seq("ATGC"), id="seq3", letter_annotations={"phred_quality": [30]*4}),
            # Высокий GC (100% GC)
            SeqRecord(Seq("GCGCGCGCGC"), id="seq4", letter_annotations={"phred_quality": [30]*10}),
        ]
        
        with open(file_path, "w") as f:
            SeqIO.write(records, f, "fastq")
            
        return file_path

    def test_fastq_file_read_write(self, sample_fastq_file, tmp_path):
        """6. Проверка, что файл читается и создается в папке filtered"""
        out_file = tmp_path / "out.fastq"
        
        stats = filter_fastq(
            source_file=str(sample_fastq_file),
            destination_file=str(out_file),
            quality_cutoff=20.0
        )
        
        expected_output_path = tmp_path / "filtered" / "out.fastq"
        
        assert expected_output_path.exists(), "Выходной файл не был создан!"
        assert stats["total_reads"] == 4
        
    def test_fastq_quality_filter(self, sample_fastq_file, tmp_path):
        """7. Фильтрация по качеству"""
        out_file = tmp_path / "out.fastq"
        stats = filter_fastq(
            source_file=str(sample_fastq_file),
            destination_file=str(out_file),
            quality_cutoff=25.0  # seq2 должен отвалиться
        )
        assert stats["passed_reads"] == 3
        assert stats["rejected_reads"] == 1

    def test_fastq_length_and_gc_filter(self, sample_fastq_file, tmp_path):
        """8. Фильтрация по длине и GC контенту"""
        out_file = tmp_path / "out.fastq"
        stats = filter_fastq(
            source_file=str(sample_fastq_file),
            destination_file=str(out_file),
            length_constraints=(8, 12), # seq3 отвалится
            gc_constraints=(0.4, 0.6)   # seq4 отвалится
        )
        # Останется только seq1 (seq2 проходит по длине/GC, но качество здесь по умолчанию 0)
        assert stats["passed_reads"] == 2 # seq1 и seq2 проходят