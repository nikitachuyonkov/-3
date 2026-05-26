import cv2
import numpy as np
import time
import os
import glob

# --- НАСТРОЙКИ ---
CASCADE_FACE = 'haarcascade_frontalface_default.xml'
CASCADE_EYE = 'haarcascade_eye.xml'
CASCADE_SMILE = 'haarcascade_smile.xml'
MASK_PATH = 'mask.png'

DEFAULT_SCALE = 1.1
DEFAULT_MIN_NEIGHBORS = 5

def load_cascades():
    face = cv2.CascadeClassifier(CASCADE_FACE)
    eye = cv2.CascadeClassifier(CASCADE_EYE)
    smile = cv2.CascadeClassifier(CASCADE_SMILE)
    
    if face.empty():
        face = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_FACE)
    if eye.empty():
        eye = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_EYE)
    if smile.empty():
        smile = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_SMILE)

    if face.empty() or eye.empty():
        raise FileNotFoundError("Не удалось загрузить каскады ни из папки проекта, ни из системы.")
    return face, eye, smile

def load_mask():
    if os.path.exists(MASK_PATH):
        mask = cv2.imread(MASK_PATH, cv2.IMREAD_UNCHANGED)
        if mask is not None and mask.shape[2] == 4:
            return mask
    print("Маска не найдена или не имеет альфа-канала. Будет использован прямоугольник-заглушка.")
    return None

def apply_ar_mask(frame, x, y, w, h, mask_img):
    if mask_img is not None:
        mask_resized = cv2.resize(mask_img, (w, h))
        b, g, r, a = cv2.split(mask_resized)
        mask_rgb = cv2.merge((b, g, r))
        mask_alpha = a.astype(float) / 255.0

        roi = frame[y:y+h, x:x+w]
        if roi.shape[:2] != mask_rgb.shape[:2]:
            return

        for c in range(3):
            roi[:, :, c] = roi[:, :, c] * (1.0 - mask_alpha) + mask_rgb[:, :, c] * mask_alpha
    else:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "MASK", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def process_static_images(face_cascade, eye_cascade, mask_img):
    print("\n--- ПУНКТ 2: ОБРАБОТКА СТАТИЧНЫХ ФОТО ---")
    image_files = glob.glob("*.jpg") + glob.glob("*.png")
    if not image_files:
        print("Не найдено изображений в текущей папке. Положите 5-7 фото рядом со скриптом.")
        return

    output_dir = "results_static"
    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=DEFAULT_SCALE, minNeighbors=DEFAULT_MIN_NEIGHBORS, minSize=(30, 30))

        for (x, y, w, h) in faces:
            apply_ar_mask(img, x, y, w, h, mask_img)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.15, minNeighbors=8, minSize=(15, 15))
            # Оставляем максимум 2 самых крупных глаза, чтобы избежать вложенных квадратов
            if len(eyes) > 2:
                eyes = sorted(eyes, key=lambda box: box[2] * box[3], reverse=True)[:2]
                
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        out_path = os.path.join(output_dir, f"detected_{os.path.basename(img_path)}")
        cv2.imwrite(out_path, img)
        print(f"Обработано: {img_path} -> {out_path}")
    print("Статичные фото сохранены в папке 'results_static'\n")

def run_webcam(face_cascade, eye_cascade, smile_cascade, mask_img):
    print("\n--- ПУНКТЫ 3, 5, 6: ВЕБ-КАМЕРА В РЕАЛЬНОМ ВРЕМЕНИ ---")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Не удалось открыть камеру")

    prev_time = time.time()
    fps_list = []
    prev_count = 0
    notify_text = ""
    notify_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 🔹 Улучшенная детекция лиц: добавлен minSize и чуть строже minNeighbors
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30)
            )
            current_count = len(faces)

            if current_count != prev_count:
                diff = current_count - prev_count
                notify_text = f"+{diff}" if diff > 0 else f"{diff}"
                notify_time = time.time()
            prev_count = current_count

            if time.time() - notify_time < 1.5 and notify_text:
                cv2.putText(frame, notify_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(frame, f"People: {current_count}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            for (x, y, w, h) in faces:
                apply_ar_mask(frame, x, y, w, h, mask_img)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                # 🔹 Улучшенная детекция глаз: строже параметры + фильтр на 2 глаза
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.15, minNeighbors=10, minSize=(15, 15))
                if len(eyes) > 2:
                    eyes = sorted(eyes, key=lambda box: box[2] * box[3], reverse=True)[:2]

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

                smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)

            cv2.putText(frame, f"FPS: {int(avg_fps)}", (frame.shape[1] - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Face Detection Lab", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n[INFO] Программа остановлена пользователем.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def test_parameters(face_cascade, eye_cascade, mask_img):
    print("\n--- ПУНКТ 4: ИССЛЕДОВАНИЕ ПАРАМЕТРОВ ---")
    print("Направьте камеру на статичную сцену с 1-2 лицами.")
    print("Нажмите 'q' в окне камеры для запуска автоматического теста...\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.putText(frame, "Press 'q' to start parameter test", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Test Ready", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n[INFO] Остановка...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    cap = cv2.VideoCapture(0)
    scales = [1.05, 1.1, 1.3]
    neighbors = [2, 5, 10]

    print(f"{'Scale':<8} | {'MinN':<6} | {'Avg FPS':<8} | {'Avg Faces':<10}")
    print("-" * 40)

    for s in scales:
        for n in neighbors:
            fps_vals = []
            face_counts = []
            start = time.time()

            while time.time() - start < 3.0:
                ret, frame = cap.read()
                if not ret: break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                t0 = time.time()
                faces = face_cascade.detectMultiScale(gray, scaleFactor=s, minNeighbors=n, minSize=(30, 30))
                dt = time.time() - t0

                fps_vals.append(1.0/dt if dt > 0 else 0)
                face_counts.append(len(faces))

            avg_fps = sum(fps_vals)/len(fps_vals)
            avg_faces = sum(face_counts)/len(face_counts)
            print(f"{s:<8.2f} | {n:<6} | {avg_fps:<8.1f} | {avg_faces:<10.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nТест завершён. Скопируйте таблицу в отчёт.")

if __name__ == "__main__":
    try:
        face_casc, eye_casc, smile_casc = load_cascades()
        mask = load_mask()

        print("Выберите режим:")
        print("1 - Обработка статичных фото (Пункт 2)")
        print("2 - Веб-камера в реальном времени (Пункты 3, 5, 6)")
        print("3 - Тест параметров (Пункт 4)")
        choice = input("Введите номер (1/2/3): ").strip()

        if choice == "1":
            process_static_images(face_casc, eye_casc, mask)
        elif choice == "2":
            run_webcam(face_casc, eye_casc, smile_casc, mask)
        elif choice == "3":
            test_parameters(face_casc, eye_casc, mask)
        else:
            print("Неверный выбор.")
    except Exception as e:
        print(f"Ошибка: {e}")