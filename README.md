# bioinf_galka_utilits #
В главном скрипте лежит две функции: filter_fastq.py и run_dna_rna_tools.py 
## filter_fastq.py ##
Берёт на вход 4 аргумента:

`seqs` - словарь данных, в котором лежат: название фрагмента прочитания, сама нуклеотидная последовательность и строка качества прочтений.

`gc_bounds` - интервал GC состава (в процентах) для фильтрации по-умолчанию равен (0, 100), также может подаваться одно число - верхняя граница интервала GC состава.

`length_bounds` - интервал длины для фильтрации, всё аналогично gc_bounds, но по-умолчанию равен (0, 2**32).

`quality_threshold` - пороговое значение среднего качества рида для фильтрации, по-умолчанию равно 0.


## run_dna_rna_tools.py ##
 Берёт на вход `*dna_tool` - нуклеотидные последовательности в виде строк и операцию, которую нужно провести с этими последовательностями

### Screenshots of flake8 and pytest ###
![flake8_bioinf_multytool.png](https://github.com/Nigalka/bioinf_galka_utilits/blob/HW4NiGalka/flake8_bioinf_multytool.png)
![pytest_bioinf_multytool.png](https://github.com/Nigalka/bioinf_galka_utilits/blob/HW4NiGalka/pytest_bioinf_multytool.png)

P. S. Я пытался, как-то поменять версию питона на котором будет проводить тест pytest, но не получилось
