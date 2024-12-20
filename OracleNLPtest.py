import cx_Oracle
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from fuzzywuzzy import process
from sqlalchemy import create_engine

# 1. Kết nối cơ sở dữ liệu
def connect_to_database():
    try:
        dsn = cx_Oracle.makedsn('192.169.3.197', 1521, service_name='reis')
        engine = create_engine(f'oracle+cx_oracle://EOFFICE_2017:123456@{dsn}')
        connection = engine.connect()
        print("Kết nối Oracle thành công.")
        return connection
    except Exception as e:
        print("Có lỗi khi kết nối cơ sở dữ liệu:", e)
        return None

# 2. Lấy dữ liệu từ cơ sở dữ liệu
def fetch_data_with_limit(connection, offset=0, limit=1000):
    try:
        query = f"""
        SELECT * FROM (
            SELECT ID, TRICHYEU, NGAYKY, CV_KHAN, ID_LOAICV, ROWNUM AS RN
            FROM CONG_VAN
        )
        WHERE RN > {offset} AND RN <= {offset + limit}
        """
        data = pd.read_sql(query, con=connection)
        if data.empty:
            print(f"Không tìm thấy dữ liệu ở trang hiện tại (offset={offset}).")
            return None

        data.columns = [col.strip().upper() for col in data.columns]
        return data
    except Exception as e:
        print(f"Có lỗi khi truy vấn dữ liệu: {e}")
        return None

# 3. Huấn luyện và tải mô hình Logistic Regression
def load_or_train_model(data):
    model_path = 'logistic_regression.model'
    vectorizer_path = 'tfidf_vectorizer.pkl'

    # Chuẩn bị dữ liệu
    X = data['TRICHYEU'].fillna('').astype(str)  # Văn bản
    y = data['ID_LOAICV']  # Nhãn
    # Tách dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        # Tải mô hình và vectorizer
        logistic_model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print(f"Đã tải mô hình từ '{model_path}' và vectorizer từ '{vectorizer_path}'.")
    except FileNotFoundError:
        # Huấn luyện mô hình mới
        print("Mô hình hoặc vectorizer chưa tồn tại. Đang huấn luyện mới...")
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        logistic_model = LogisticRegression(max_iter=1000, random_state=42)
        logistic_model.fit(X_train_tfidf, y_train)

        # Lưu mô hình và vectorizer
        joblib.dump(logistic_model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Đã lưu mô hình vào '{model_path}' và vectorizer vào '{vectorizer_path}'.")

    # Đánh giá mô hình
    X_test_tfidf = vectorizer.transform(X_test)
    predictions = logistic_model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nĐộ chính xác của mô hình: {accuracy * 100:.2f}%")
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, predictions))


    return logistic_model, vectorizer

# 4. Tìm kiếm gần đúng với mô hình
def search_keyword_with_model(data, keyword, model, vectorizer):
    try:
        matches = process.extract(keyword, data['TRICHYEU'].astype(str), limit=5)
        print("\nKết quả tìm kiếm (gần đúng):")
        if not matches:
            print("Không tìm thấy kết quả phù hợp.")
            return

        for match in matches:
            matched_value = match[0]
            row = data[data['TRICHYEU'] == matched_value].iloc[0]
            print(f"ID: {row['ID']}, TRICHYEU: {row['TRICHYEU']}, NGAYKY: {row['NGAYKY']}, ID_LOAICV: {row['ID_LOAICV']}")

        keyword_tfidf = vectorizer.transform([keyword])
        predicted_label = model.predict(keyword_tfidf)[0]
        print(f"\nDự đoán nhãn ID_LOAICV cho từ khóa '{keyword}': {predicted_label}")
    except Exception as e:
        print("Có lỗi trong khi tìm kiếm:", str(e))

# 5. Tìm kiếm theo ID_LOAICV
def search_by_id_loaicv_with_model(connection, id_loaicv, model, vectorizer, limit=1000):
    offset = 0
    while True:
        data = fetch_data_with_limit(connection, offset=offset, limit=limit)

        if data is None or data.empty:
            print("Không còn dữ liệu để hiển thị.")
            break
        filtered_data = data[data['ID_LOAICV'] == id_loaicv]
        if filtered_data.empty:
            print(f"Không tìm thấy dữ liệu với ID_LOAICV = {id_loaicv}.")
        else:
            print(f"\nKết quả cho ID_LOAICV = {id_loaicv}, từ dòng {offset + 1} đến {offset + len(data)}:")
            for _, row in filtered_data.iterrows():
                print(f"ID: {row['ID']}, TRICHYEU: {row['TRICHYEU']}, NGAYKY: {row['NGAYKY']}, CV_KHAN: {row['CV_KHAN']}, ID_LOAICV: {row['ID_LOAICV']}")
        choice = input("Nhập 'tiep' để tải thêm, hoặc bất kỳ phím nào để quay lại: ").strip().lower()
        if choice != 'tiep':
            break
        offset += limit

# 6. Hàm chính
def main():
    connection = connect_to_database()
    if connection is None:
        return

    # Lấy dữ liệu ban đầu để huấn luyện mô hình
    initial_data = fetch_data_with_limit(connection, offset=0)
    if initial_data is None or len(set(initial_data['ID_LOAICV'])) < 2:
        print("Dữ liệu không đủ điều kiện để huấn luyện mô hình (cần ít nhất hai nhãn khác nhau).")
        return

    model, vectorizer = load_or_train_model(initial_data)
    if model is None or vectorizer is None:
        return

    try:
        while True:
            choice = input("\nNhập '1' để tìm kiếm theo ID_LOAICV, '2' để tìm kiếm gần đúng, hoặc 'thoat' để thoát: ")
            if choice.lower() == 'thoat':
                break
            elif choice == '1':
                try:
                    id_loaicv = int(input("Nhập ID loại công văn để tìm kiếm: "))
                    #limit = int(input("Nhập số dòng muốn hiển thị mỗi lần (chỉ nhập số nguyên): "))
                    search_by_id_loaicv_with_model(connection, id_loaicv, model, vectorizer)
                except ValueError:
                    print("ID_LOAICV và số dòng phải là số nguyên.")
            elif choice == '2':
                keyword = input("Nhập từ khóa để tìm kiếm: ")
                search_keyword_with_model(initial_data, keyword, model, vectorizer)
            else:
                print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
    finally:
        connection.close()
        print("Ngắt kết nối.")

if __name__ == "__main__":
    main()