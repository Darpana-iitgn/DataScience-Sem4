# DataScience-Sem4
Interesting projects for CS308



# Project Structure
datasets  ---> Contains all the datasets
main.ipynb ---> main notebook used by viola
dockerfile---> docker configuration file


# Quick setup

1) docker build -t my-voila-app .
2) docker run -p 8866:8866 my-voila-app
3) visit localhost:8866


# Using jupyter book
Just run 
bash makebook.sh  it will create the index.html in docs which will be rendered by github pages