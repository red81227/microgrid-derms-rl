@echo off
cd ..
mkdir   \tmp \data\ncku-derms \data\ncku-derms\logs

:: 授予写入权限
icacls \data\ncku-derms\ /grant Everyone:(OI)(CI)F

icacls \tmp /grant Everyone:(OI)(CI)F