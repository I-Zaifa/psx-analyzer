"""Generate PWA icon PNGs from the SVG favicon."""
import struct
import zlib
import os

def create_png(width, height, filepath):
    """Create a simple PNG icon with teal background and chart line."""
    # Create RGBA pixel data
    pixels = []
    bg = (207, 181, 59, 255)  # Coin Gold
    white = (255, 255, 255, 255)
    
    # Sharp corners (no radius)
    radius = 0
    
    # Chart line points (normalized to icon size)
    chart_points = [
        (0.25, 0.6875),
        (0.375, 0.4375),
        (0.5, 0.5625),
        (0.625, 0.3125),
        (0.75, 0.5),
    ]
    chart_px = [(int(x * width), int(y * height)) for x, y in chart_points]
    
    for y in range(height):
        row = []
        for x in range(width):
            # Rounded corners
            in_rect = True
            for cx, cy in [(radius, radius), (width - radius - 1, radius),
                           (radius, height - radius - 1), (width - radius - 1, height - radius - 1)]:
                if ((x < radius or x >= width - radius) and (y < radius or y >= height - radius)):
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy > radius * radius:
                        in_rect = False
                        break
            
            if not in_rect:
                row.extend([0, 0, 0, 0])  # Transparent
                continue
            
            # Check if pixel is on the chart line
            on_line = False
            line_width = max(2, width // 12)
            
            for i in range(len(chart_px) - 1):
                x1, y1 = chart_px[i]
                x2, y2 = chart_px[i + 1]
                if min(x1, x2) - line_width <= x <= max(x1, x2) + line_width:
                    if x2 != x1:
                        t = (x - x1) / (x2 - x1)
                        t = max(0, min(1, t))
                        ly = y1 + t * (y2 - y1)
                        if abs(y - ly) <= line_width:
                            on_line = True
                            break
            
            # Circle at end point
            ex, ey = chart_px[-1]
            circle_r = max(2, width // 16)
            if (x - ex) ** 2 + (y - ey) ** 2 <= circle_r ** 2:
                on_line = True
            
            if on_line:
                row.extend(white)
            else:
                row.extend(bg)
        
        pixels.append(bytes(row))
    
    # Build PNG
    def make_chunk(chunk_type, data):
        chunk = chunk_type + data
        return struct.pack('>I', len(data)) + chunk + struct.pack('>I', zlib.crc32(chunk) & 0xffffffff)
    
    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'
    
    # IHDR
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
    ihdr = make_chunk(b'IHDR', ihdr_data)
    
    # IDAT
    raw_data = b''
    for row in pixels:
        raw_data += b'\x00' + row  # Filter: None
    
    compressed = zlib.compress(raw_data, 9)
    idat = make_chunk(b'IDAT', compressed)
    
    # IEND
    iend = make_chunk(b'IEND', b'')
    
    with open(filepath, 'wb') as f:
        f.write(signature + ihdr + idat + iend)
    
    print(f"Created {filepath} ({width}x{height})")


if __name__ == "__main__":
    icons_dir = os.path.join(os.path.dirname(__file__), "app", "static", "icons")
    os.makedirs(icons_dir, exist_ok=True)
    create_png(192, 192, os.path.join(icons_dir, "icon-192.png"))
    create_png(512, 512, os.path.join(icons_dir, "icon-512.png"))
    print("Icons generated!")
