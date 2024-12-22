import numpy as np
import cv2

class ImageBuffer:
    def __init__(self, buffer_size, frame_shape):
        """
        :param buffer_size: Jumlah frame yang ingin disimpan di buffer.
        :param frame_shape: Dimensi frame (height, width, channels).
        """
        self.buffer_size = buffer_size
        self.buffer = np.zeros((buffer_size, *frame_shape), dtype=np.float32)  # Buffer berisi frame-frame.
        self.index = 0  # Indeks untuk memasukkan frame baru.
        self.filled = 0  # Counter untuk mengetahui berapa banyak frame yang sudah diisi di buffer.

    def add_frame(self, frame):
        """Menambahkan frame baru ke buffer."""
        self.buffer[self.index] = frame.astype(np.float32)  # Menambahkan frame ke buffer.
        self.index = (self.index + 1) % self.buffer_size  # Perbarui indeks secara melingkar.
        self.filled = min(self.filled + 1, self.buffer_size)  # Update jumlah frame yang terisi di buffer.

    def get_averaged_frame(self):
        """Mengembalikan frame yang dirata-rata dari buffer."""
        if self.filled == 0:
            raise ValueError("Buffer masih kosong!")
        return (np.sum(self.buffer[:self.filled], axis=0) / self.filled).astype(np.uint8)  # Rata-rata frame.

# Contoh penggunaan:
if __name__ == "__main__":
    # Inisialisasi video capture dan buffer.
    cap = cv2.VideoCapture('sweeping.mp4')  # Ambil video dari kamera.
    ret, frame = cap.read()
    if not ret:
        print("Gagal menangkap frame dari kamera.")
        cap.release()
        exit()

    frame_shape = frame.shape  # Dimensi frame pertama.
    buffer_size = 5  # Misal buffer menyimpan 5 frame.
    buffer = ImageBuffer(buffer_size, frame_shape)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal menangkap frame.")
            break

        buffer.add_frame(frame)  # Tambahkan frame ke buffer.

        try:
            averaged_frame = buffer.get_averaged_frame()  # Dapatkan frame rata-rata.
            resized_frame = cv2.resize(averaged_frame, (320, 240))  # Resize ke resolusi lebih kecil.

            # Tampilkan frame asli dan hasil averaging.
            cv2.imshow("Original Frame", frame)
            cv2.imshow("Averaged Frame", resized_frame)
        except ValueError:
            pass  # Tunggu hingga buffer terisi.

        # Keluar jika tombol 'q' ditekan.
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
