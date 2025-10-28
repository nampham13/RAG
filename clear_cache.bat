@echo off
REM ================================================
REM Script xóa tất cả thư mục __pycache__ trong dự án
REM ================================================
 
echo 🔄 Đang tìm và xóa tất cả thư mục __pycache__ ...
 
REM Dùng for /r để duyệt toàn bộ thư mục con
for /d /r %%i in (__pycache__) do (
    echo 🗑️ Xóa: %%i
    rmdir /s /q "%%i"
)
 
echo ✅ Đã xóa toàn bộ __pycache__ trong dự án!
 