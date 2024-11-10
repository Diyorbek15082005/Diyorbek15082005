import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tkinter as tk
from tkinter import ttk

# Ma'lumotlar to'plami yaratish (misol sifatida)
data = {
    'data_size': ['small', 'large', 'small', 'moderate', 'large'],
    'complexity': ['high', 'low', 'low', 'high', 'moderate'],
    'speed': ['moderate', 'high', 'very high', 'moderate', 'high'],
    'recommended_db': ['Relational', 'NoSQL', 'In-Memory', 'Object-Oriented', 'NoSQL']
}

# Pandas DataFrame ga o‘tkazish
df = pd.DataFrame(data)

# Xususiyatlar va nishonni ajratish
X = pd.get_dummies(df[['data_size', 'complexity', 'speed']])
y = pd.get_dummies(df['recommended_db'])

# Ma'lumotlar to‘plamining numpy formatiga aylantirilishi
X = np.array(X)
y = np.array(y)

# Neyron tarmoq modelini yaratish
model = Sequential([
    Dense(16, input_shape=(X.shape[1],), activation='relu'),
    Dense(16, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

# Modelni kompilatsiya qilish
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modelni o‘rgatish
model.fit(X, y, epochs=50, verbose=0)

# Tavsiya olish funksiyasi
def get_recommendation():
    # Foydalanuvchi kiritmalarini olish
    data_size = data_size_var.get()
    complexity = complexity_var.get()
    speed = speed_var.get()

    # Kiritmalarni model uchun kerakli formatga aylantirish
    input_data = pd.DataFrame([[data_size, complexity, speed]], columns=['data_size', 'complexity', 'speed'])
    input_data = pd.get_dummies(input_data).reindex(columns=df.columns[:-1], fill_value=0)
    input_data = np.array(input_data)

    # Model yordamida tavsiya olish
    prediction = model.predict(input_data)
    recommended_db = y.columns[np.argmax(prediction)]

    # Natijani interfeysda ko'rsatish
    result_label.config(text=f"Tavsiya etilgan saqlash usuli: {recommended_db}")

# Interfeysni yaratish
app = tk.Tk()
app.title("Bilimlar Bazasi Saqlash Usullari Tavsiya Dasturi")
app.geometry("400x300")

# Parametrlarni tanlash uchun interfeys elementlari
ttk.Label(app, text="Ma'lumot hajmi").pack(pady=5)
data_size_var = tk.StringVar()
data_size_combo = ttk.Combobox(app, textvariable=data_size_var)
data_size_combo['values'] = ('small', 'moderate', 'large')
data_size_combo.pack()

ttk.Label(app, text="Ma'lumotlar murakkabligi").pack(pady=5)
complexity_var = tk.StringVar()
complexity_combo = ttk.Combobox(app, textvariable=complexity_var)
complexity_combo['values'] = ('low', 'moderate', 'high')
complexity_combo.pack()

ttk.Label(app, text="Ishlash tezligi").pack(pady=5)
speed_var = tk.StringVar()
speed_combo = ttk.Combobox(app, textvariable=speed_var)
speed_combo['values'] = ('low', 'moderate', 'high', 'very high')
speed_combo.pack()

# Natijani ko'rsatish tugmasi
ttk.Button(app, text="Tavsiya olish", command=get_recommendation).pack(pady=20)

# Natija uchun label
result_label = ttk.Label(app, text="")
result_label.pack(pady=5)

# Dastur ishga tushirilishi
app.mainloop()