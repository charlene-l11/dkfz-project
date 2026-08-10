from pathlib import Path
from PIL import Image, ImageDraw

paths = sorted(Path("tmp/docx_render").glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
thumbs = []
for path in paths:
    image = Image.open(path).convert("RGB")
    image.thumbnail((380, 492))
    canvas = Image.new("RGB", (400, 530), "white")
    canvas.paste(image, ((400-image.width)//2, 24))
    ImageDraw.Draw(canvas).text((12, 6), path.stem, fill="black")
    thumbs.append(canvas)
cols = 3
rows = (len(thumbs) + cols - 1) // cols
sheet = Image.new("RGB", (cols*400, rows*530), "#dddddd")
for i, image in enumerate(thumbs):
    sheet.paste(image, ((i % cols)*400, (i // cols)*530))
sheet.save("tmp/docx_render/contact-sheet.png")
