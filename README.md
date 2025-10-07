# 🧠 Simulating a Brain Circuit in Python  
*A hands-on scientific computing project inspired by computational neuroscience.*

---

## 📘 Project Overview
This project implements the **Izhikevich spiking neuron model** — one of the most computationally efficient and biologically realistic models of neuron firing dynamics.

It is part of my learning journey in the Udemy course **[Master Python Programming by Solving Scientific Projects](https://www.udemy.com/course/python-scientific-x/learn/)** by **Mike X Cohen**.  
Through this course, I’m exploring how Python can be used to simulate real-world scientific and engineering systems.

---

## 🧩 Model Reference
**Paper:** *Eugene M. Izhikevich (2003), “Simple Model of Spiking Neurons,” IEEE Transactions on Neural Networks, Vol. 14, No. 6.*  
This model combines the biological plausibility of the Hodgkin–Huxley equations with the computational simplicity of integrate-and-fire neurons.

**Core Equations:**
\[
v' = 0.04v^2 + 5v + 140 - u + I
\]
\[
u' = a(bv - u)
\]
with reset conditions:
\[
\text{if } v \ge 30: \quad v \leftarrow c,\; u \leftarrow u + d
\]

---

## 💻 Simulation Details
The simulation reproduces cortical neuron spiking behavior using Python and `matplotlib`.  
Each spike corresponds to the firing of a single neuron under constant input current.

**Key Parameters:**
| Parameter | Description | Typical Value |
|------------|-------------|----------------|
| a | Time scale of recovery variable | 0.02 |
| b | Sensitivity of recovery variable | 0.2 |
| c | After-spike reset of membrane potential | -65 mV |
| d | After-spike reset of recovery variable | 8 |

---

## 🧠 Sample Code Snippet
```python
import numpy as np
import matplotlib.pyplot as plt

a, b, c, d = 0.02, 0.2, -65, 8
v, u = -65, b * v
tau = 0.25
V = []

for t in range(1000):
    I = 10 if 100 < t < 900 else 0
    v += tau * (0.04*v**2 + 5*v + 140 - u + I)
    u += tau * a * (b*v - u)
    if v >= 30:
        V.append(30)
        v, u = c, u + d
    else:
        V.append(v)

plt.plot(V)
plt.title("Izhikevich Neuron Model Simulation")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.show()

