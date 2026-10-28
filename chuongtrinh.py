

# BUOC 1. KHAI BAO THU VIEN
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Lay thu muc chua chuong trinh de doc CSV dung du mo bang VS Code
thu_muc = Path(__file__).resolve().parent
thu_muc_ket_qua = thu_muc / "ket_qua"
thu_muc_ket_qua.mkdir(exist_ok=True)

# BUOC 2. DOC FILE CSV
# utf-8-sig giup doc dung tieng Viet trong file CSV.
df = pd.read_csv(thu_muc / "du_lieu_sieu_thi.csv", encoding="utf-8-sig")
print("So dong, so cot:", df.shape)
print("So o bi thieu:", df.isnull().sum().sum())
print("So dong trung nhau:", df.duplicated().sum())

# BUOC 3. CHON HAI COT DE PHAN CUM
cot_doanh_thu = "doanh thu (đã tính thuế)"
cot_so_luong = "so_luong"

# Chuyen sang dang so, bo nhung dong khong hop le (neu co).
df[cot_doanh_thu] = pd.to_numeric(df[cot_doanh_thu], errors="coerce")
df[cot_so_luong] = pd.to_numeric(df[cot_so_luong], errors="coerce")
df = df.dropna(subset=[cot_doanh_thu, cot_so_luong]).copy()
df = df[(df[cot_doanh_thu] > 0) & (df[cot_so_luong] > 0)].copy()

print("\nTHONG KE CO BAN")
print(df[[cot_doanh_thu, cot_so_luong]].describe().round(2))

# BUOC 4. CHUAN HOA DU LIEU
# Vi doanh thu va so luong co don vi khac nhau nen phai chuan hoa.
X = df[[cot_doanh_thu, cot_so_luong]]
scaler = StandardScaler()
X_chuan_hoa = scaler.fit_transform(X)

# BUOC 5. THU K TU 2 DEN 6 DE CHON SO CUM
cac_k = [2, 3, 4, 5, 6]
cac_inertia = []
cac_silhouette = []

for k in cac_k:
    mo_hinh = KMeans(n_clusters=k, random_state=42, n_init=10)
    nhan_cum = mo_hinh.fit_predict(X_chuan_hoa)
    cac_inertia.append(mo_hinh.inertia_)
    cac_silhouette.append(silhouette_score(X_chuan_hoa, nhan_cum))

# Chon K co Silhouette Score cao nhat trong cac gia tri da thu.
k_tot_nhat = cac_k[cac_silhouette.index(max(cac_silhouette))]
print("\nDANH GIA SO CUM")
for i in range(len(cac_k)):
    print("K =", cac_k[i], "| Inertia =", round(cac_inertia[i], 3),
          "| Silhouette =", round(cac_silhouette[i], 3))
print("K duoc chon:", k_tot_nhat)

# BUOC 6. CHAY K-MEANS VA GAN NHAN CUM
kmeans = KMeans(n_clusters=k_tot_nhat, random_state=42, n_init=10)
df["cum_kmeans"] = kmeans.fit_predict(X_chuan_hoa)

# In thong ke cua tung cum.
thong_ke_cum = df.groupby("cum_kmeans")[[cot_doanh_thu, cot_so_luong]].agg(
    ["count", "mean", "sum"]
)
print("\nTHONG KE THEO CUM")
print(thong_ke_cum.round(2))

# BUOC 7. VE DU 5 BIEU DO
# Hinh 01: Tong doanh thu theo nhom hang.
doanh_thu_nhom = df.groupby("nhom_hang")[cot_doanh_thu].sum().sort_values()
plt.figure(figsize=(10, 5))
doanh_thu_nhom.plot(kind="barh")
plt.title("01. Tong doanh thu theo nhom hang")
plt.xlabel("Doanh thu da tinh thue")
plt.tight_layout()
plt.savefig(thu_muc_ket_qua / "01_doanh_thu_theo_nhom_hang.png", dpi=150)
plt.close()

# Hinh 02: Phan phoi doanh thu giao dich.
plt.figure(figsize=(8, 5))
plt.hist(df[cot_doanh_thu], bins=20, edgecolor="black")
plt.title("02. Phan phoi doanh thu")
plt.xlabel("Doanh thu moi giao dich")
plt.ylabel("So giao dich")
plt.tight_layout()
plt.savefig(thu_muc_ket_qua / "02_phan_phoi_doanh_thu.png", dpi=150)
plt.close()

# Hinh 03: Elbow - xem Inertia thay doi khi tang K.
plt.figure(figsize=(7, 5))
plt.plot(cac_k, cac_inertia, marker="o")
plt.title("03. Elbow - Inertia theo K")
plt.xlabel("So cum K")
plt.ylabel("Inertia")
plt.xticks(cac_k)
plt.grid(True)
plt.tight_layout()
plt.savefig(thu_muc_ket_qua / "03_bieu_do_elbow.png", dpi=150)
plt.close()

# Hinh 04: Silhouette Score - danh gia so cum.
plt.figure(figsize=(7, 5))
plt.plot(cac_k, cac_silhouette, marker="o")
plt.title("04. Silhouette Score theo K")
plt.xlabel("So cum K")
plt.ylabel("Silhouette Score")
plt.xticks(cac_k)
plt.grid(True)
plt.tight_layout()
plt.savefig(thu_muc_ket_qua / "04_diem_silhouette.png", dpi=150)
plt.close()

# Hinh 05: Ket qua phan cum, moi cham la mot giao dich.
plt.figure(figsize=(9, 6))
for cum in sorted(df["cum_kmeans"].unique()):
    du_lieu_cum = df[df["cum_kmeans"] == cum]
    plt.scatter(du_lieu_cum[cot_doanh_thu], du_lieu_cum[cot_so_luong],
                label="Cum " + str(cum), alpha=0.6, s=22)

# Dua tam cum tu du lieu chuan hoa ve gia tri goc de ve dung truc.
tam_cum = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(tam_cum[:, 0], tam_cum[:, 1], marker="X", s=180,
            c="black", label="Tam cum")
plt.title("05. Ket qua phan cum K-Means")
plt.xlabel("Doanh thu da tinh thue")
plt.ylabel("So luong")
plt.legend()
plt.tight_layout()
plt.savefig(thu_muc_ket_qua / "05_bieu_do_phan_cum.png", dpi=150)
plt.close()

# BUOC 8. LUU KET QUA RA CSV (KHONG DUNG EXCEL)
df.to_csv(thu_muc_ket_qua / "du_lieu_da_phan_cum.csv",
          index=False, encoding="utf-8-sig")

bang_k = pd.DataFrame({"K": cac_k, "Inertia": cac_inertia,
                       "Silhouette": cac_silhouette})
bang_k.to_csv(thu_muc_ket_qua / "danh_gia_k.csv",
                index=False, encoding="utf-8-sig")

print("\nDA HOAN THANH! Ket qua nam trong thu muc: ket_qua")
