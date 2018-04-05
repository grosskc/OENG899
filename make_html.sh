#!/bin/bash

# sed 's/^```python/```{python, engine.path="\/anaconda3\/bin\/python"}/' $1 > ${1%%.*}.Rmd
# rCmd='library(rmarkdown); rmarkdown::render("'
# rCmd=${rCmd}${1%%.*}".Rmd"
# rCmd=${rCmd}'", "html_document")'
# rCmd="'"$rCmd"'"
# rCmd="/Library/Frameworks/R.framework/Versions/3.4/Resources/bin/Rscript -e "${rCmd}
# echo ${rCmd}
# eval ${rCmd}

# /anaconda3/bin/pandoc +RTS -K512m -RTS LWIR_HSI_model.utf8.md --to html4 --from markdown+autolink_bare_uris+ascii_identifiers+tex_math_single_backslash --output LWIR_HSI_model.html --smart --email-obfuscation none --self-contained --standalone --section-divs --template /Library/Frameworks/R.framework/Versions/3.4/Resources/library/rmarkdown/rmd/h/default.html --no-highlight --variable highlightjs=1 --variable 'theme:bootstrap' --include-in-header /var/folders/36/jz538fk14kd2k2l29jg5q32w0000gn/T//Rtmp57jdWL/rmarkdown-str702d368b3c44.html --mathjax --variable 'mathjax-url:https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML' --variable code_folding=hide --variable code_menu=1

/anaconda3/bin/pandoc ${1} --output ${1%%.*}.html --to html5 --from markdown --smart --section-divs --highlight-style haddock --template pandoc.html --css pandoc.css --mathjax --variable 'mathjax-url:https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML' --variable code_folding=hide
