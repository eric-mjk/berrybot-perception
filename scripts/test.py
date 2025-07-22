from pathlib import Path

url = r"C:\Users\ericm\Desktop\BerryBot\Tests_Tests\TEST1\NaiveTraining_Results2\runs\pose\yolov8n_strawberry_continued\weights\best.pt"

path = Path(url)
print(path.as_posix())