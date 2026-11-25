# Uzdevums: 3. Realizēt trokšņu novākšanu ar Hāra vilnīšu transformāciju.
# Autors: Reinis Ričards Ramis
# Stud. apl. nr.: rr26072
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

T = 40  # slieksnis: detaļas ar mazāku absolūto vērtību tiek nonullētas

def forward(x):
    a, b = x[0::2], x[1::2]
    return np.vstack(((a + b) / 2, (a - b) / 2))  # augšā l, apakšā h

def inverse(x):
    n = x.shape[0] // 2
    l, h = x[:n], x[n:]
    out = np.empty_like(x)
    out[0::2] = l + h
    out[1::2] = l - h
    return out


# 1. Ielādē bildi, pārtaisa par matricu
img = np.array(Image.open("example_noisy.jpg").convert("L"), dtype=float)

# 2. Katrai rindai izsauc l = (a + b)/2, h = (a – b)/2, kur a un b ir divi blakus esoši pikseļi
y = forward(img.T).T

# 3. Katrai kolonnai izdara to pašu
y = forward(y)

# 4. Nonullē h objektus ar vērtībām zem sliekšņa T
h2, w2 = y.shape[0] // 2, y.shape[1] // 2
ll = y[:h2, :w2].copy()
y[np.abs(y) < T] = 0
y[:h2, :w2] = ll

# 5. Katram l,h pārim sauc a = l + h, b = l – h, lai izveidotu jaunu matricu ar modificēto bildi
result = inverse(y)
result = inverse(result.T).T

# 6. Attēlo bildi
result = np.clip(result, 0, 255)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Trokšņainā bilde")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray", vmin=0, vmax=255)
plt.title(f"Pēc trokšņu novākšanas (T = {T})")
plt.axis("off")
plt.show()