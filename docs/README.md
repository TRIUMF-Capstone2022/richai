# Project reports

This directory contains the files related to the project proposal and project final report, as well as images that are used in the root directory README file.

## Project proposal

If you would like to re-generate the project proposal PDF report, first ensure that you have [`R`](https://www.r-project.org/) installed, and then you can run the following command from the root directory of this project:

`Rscript -e 'rmarkdown::render(".docs/proposal/report/proposal.Rmd")'`

## Final report

The report is hosted as a [Jupyter Book](https://jupyterbook.org/en/stable/intro.html) on GitHub pages, and the underlying files that are used to build the Jupyter Book can be accessed [here](https://github.com/TRIUMF-Capstone2022/richai/tree/main/docs/final_report).  The Jupyter Book itself is built automatically via a [GitHub Actions workflow](https://github.com/TRIUMF-Capstone2022/richai/blob/main/.github/workflows/final_report.yml), which triggers if there is a push to the `main` branch of this repository that changes a file within the `richai/docs/final_report/` directory.

### Generating the Jupyter Book locally

If you would like to build the Jupyter Book locally, first ensure that you have `python` and `jupyter-book` installed, and then you can run the following commands from the root directory of the project:

```
# to copy the notebooks into the final report folder
mkdir docs/final_report/appendix/notebooks
cp -r notebooks/** docs/final_report/appendix/notebooks

# to build the jupyter book
jupyter-book build --all final_report
```

You will then need to open a `.html` file from within the `_build/` directory that the above `jupyter-book` command generates, which will display the Jupyter Book locally in your web browser.