@echo off
REM Build script for FoodPhotoEnhancer Docker image

echo Building FoodPhotoEnhancer Docker image...
docker build -t foodphotoenhancer:latest .

if %errorlevel% neq 0 (
    echo Build failed!
    exit /b %errorlevel%
)

echo.
echo Build complete! Run the service with:
echo   docker-compose up -d
echo or
echo   docker run -d -p 8000:8000 --name foodphotoenhancer foodphotoenhancer:latest
