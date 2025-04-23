# DataScience-Sem4
Interesting projects for CS308

# Graph Stream Library ;->



# Project Structure
datasets  ---> Contains all the datasets
main.ipynb ---> main notebook used by viola // Now renamed as Writing_Assignment_Final.ipynb
dockerfile---> docker configuration file

Sketch_Algorithms_Final.ipynb  // Finalized Sketch Algorithms   Note: add the comparison with ground truth values (sir's suggestion)
Streaming_Algorithms_Final.ipynb // Finalized Sketch Algorithms Note: Add the experiments with the different values of the constants to compare how the constants affects the outputs.


# Quick setup

1) docker build -t my-voila-app .
2) docker run -p 8866:8866 my-voila-app
3) visit localhost:8866


# Using jupyter book
Just run 
bash makebook.sh  it will create the index.html in docs which will be rendered by github pages