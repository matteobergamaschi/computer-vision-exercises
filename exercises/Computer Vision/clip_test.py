import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("D:\Matteo\Desktop\Magistrale\Computer Vision\Lab\cat.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # Label probs: [[0.002237 0.006374 0.991]]


# the key idea of the paper is to use a network which is not created for distinguishing real and
# fake images (that could be bad for generalization). In particular we want to distinguish in a feature
# space that has not been learned with the intent of separating the two classes (but at the end we still want)
# to separate them

# in the paper they use a variant of the ViT-L/14, trained for the task of image and language alignment

# NEAREST NEIGHBOR: in pratica (detta molto male) devo creare 2 bank R:={Embed(r1), Embed(r2),..., Embed(rN)} e
# F:={Embed(f1), Embed(f2),..., Embed(fN)}. Una volta fatto questo devo calcolare per la mia immagine "x" da
# classificare, il suo embedding 'x' = Embed(x) e poi far la cosine similarity con i 2 banks. Infine vedere
# a quali dei due è più simile e assegnare 1-fake 0-real
# il CLIP encoder è "frozen"

# LINEAR CLASSIFICATION: si aggiunge un singolo layer con sigmoide e si traina questo nuovo classificatore con 
# una binary cross-entropy loss (more computationally and memory friendly) (io userei questo metodo)

# metodo da valutare: feddare a una cnn una log-magnitude della FFT dell'immagine, dove la magnitude è data
# da M:=sqrt(Re(F)^2+Im(F)^2) e la log-magnitude:=log(1+M)

# un'altra idea sarebbe quella di utilizzare 3 diversi approcci alla classificazione, ovvero una RGB_CNN, una
# FFT_CNN e quello basato su CLIP, per cogliere tutti i possibili pattern in modo da essere robusto per
# la classificazione di elementi nuovi o realizzati da nuovi generatori