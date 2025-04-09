jupyter-book create mynewbook
cd mynewbook
jupyter-book build .
cp -r ./_build/html/* ../docs

cd ..