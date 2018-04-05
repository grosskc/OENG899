#!/bin/bash

# sed 's/^```python/```{python, engine.path="\/anaconda3\/bin\/python"}/' $1 > ${1%%.*}.Rmd
# rCmd='library(rmarkdown); rmarkdown::render("'
# rCmd=${rCmd}${1%%.*}".Rmd"
# rCmd=${rCmd}'", "html_document")'
# rCmd="'"$rCmd"'"
# rCmd="/Library/Frameworks/R.framework/Versions/3.4/Resources/bin/Rscript -e "${rCmd}
# echo ${rCmd}
# eval ${rCmd}

/anaconda3/bin/pandoc ${1} --output ${1%%.*}.html --to html5 --from markdown --smart --section-divs --highlight-style haddock --template pandoc.html --css pandoc.css --mathjax --variable 'mathjax-url:https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'
