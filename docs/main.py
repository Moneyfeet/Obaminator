import pygame
import numpy as np
from PIL import Image

# === CONFIG ===
IMAGE_B_PATH = "Obaminator/Preset/Obama.png"
WINDOW_SIZE = (800, 800)
PIXEL_STEP = 10
FRAME_RATE = 60
TOPBAR_HEIGHT = 60

PRESETS = [
    "Obaminator/Preset/Shrek.png",
    "Obaminator/Preset/Cat.png",
    "Obaminator/Preset/Tree.png"
]

# === IMAGE LOADING ===
def load_image(path, size):
    img = Image.open(path).convert("RGB").resize(size)
    return np.array(img)

# === SETUP ===
def setup_images(img_a_path, img_b_path, w, h):
    img_a = load_image(img_a_path, (w, h))
    img_b = load_image(img_b_path, (w, h))

    y, x = np.indices((h, w))
    coords = np.stack((x.flatten(), y.flatten()), axis=1)

    def luminance(rgb):
        return 0.2126 * rgb[:,0] + 0.7152 * rgb[:,1] + 0.0722 * rgb[:,2]

    src_colors = img_a.reshape(-1, 3)
    target_colors = img_b.reshape(-1, 3)

    src_order = np.argsort(luminance(src_colors))
    tgt_order = np.argsort(luminance(target_colors))

    targets = coords[tgt_order]
    positions = coords[src_order].astype(float)
    src_colors = src_colors[src_order]

    return positions, targets, src_colors

# === UI ELEMENTS ===
class Button:
    def __init__(self, rect, text):
        self.rect = pygame.Rect(rect)
        self.text = text

    def draw(self, screen, font):
        pygame.draw.rect(screen, (50, 50, 50), self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2)

        txt = font.render(self.text, True, (255, 255, 255))
        screen.blit(txt, txt.get_rect(center=self.rect.center))

    def is_clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)

class Dropdown:
    def __init__(self, rect, options):
        self.rect = pygame.Rect(rect)
        self.options = options
        self.selected = 0
        self.open = False

    def draw(self, screen, font):
        pygame.draw.rect(screen, (50, 50, 50), self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2)

        text = self.options[self.selected].split("/")[-1]
        txt = font.render(text, True, (255, 255, 255))
        screen.blit(txt, (self.rect.x + 5, self.rect.y + 10))

        if self.open:
            for i, option in enumerate(self.options):
                r = pygame.Rect(self.rect.x, self.rect.y + (i+1)*self.rect.height,
                                self.rect.width, self.rect.height)
                pygame.draw.rect(screen, (40, 40, 40), r)
                pygame.draw.rect(screen, (200, 200, 200), r, 1)

                txt = font.render(option.split("/")[-1], True, (255,255,255))
                screen.blit(txt, (r.x + 5, r.y + 10))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.open = not self.open
                return None

            if self.open:
                for i in range(len(self.options)):
                    r = pygame.Rect(self.rect.x,
                                    self.rect.y + (i+1)*self.rect.height,
                                    self.rect.width,
                                    self.rect.height)
                    if r.collidepoint(event.pos):
                        self.selected = i
                        self.open = False
                        return self.options[i]

                self.open = False
        return None

# === MAIN ===
def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Obaminator")

    font = pygame.font.SysFont(None, 24)
    w, h = WINDOW_SIZE

    dropdown = Dropdown((10, 10, 220, 40), PRESETS)
    start_btn = Button((240, 10, 120, 40), "Start")

    # Preload state
    positions, targets, src_colors = None, None, None
    preloaded = False

    clock = pygame.time.Clock()
    running = True
    animating = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            selected = dropdown.handle_event(event)

            # 🔹 PRELOAD on dropdown select
            if selected is not None:
                animating = False
                positions, targets, src_colors = setup_images(
                    selected, IMAGE_B_PATH, w, h
                )
                preloaded = True

            # 🔹 START animation only if preloaded
            if start_btn.is_clicked(event) and preloaded:
                animating = True

        # Background
        screen.fill((20, 20, 20))

                # Animation
        if animating and positions is not None:
            delta = targets - positions
            dist = np.linalg.norm(delta, axis=1)
            mask = dist > 1

            positions[mask] += np.clip(delta[mask], -PIXEL_STEP, PIXEL_STEP)

            frame = np.zeros((h, w, 3), dtype=np.uint8)
            xi = np.clip(positions[:, 0].astype(int), 0, w - 1)
            yi = np.clip(positions[:, 1].astype(int), 0, h - 1)
            frame[yi, xi] = src_colors

            pygame.surfarray.blit_array(screen, frame.swapaxes(0, 1))

        # Show preloaded image
        elif preloaded and positions is not None:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            xi = np.clip(positions[:, 0].astype(int), 0, w - 1)
            yi = np.clip(positions[:, 1].astype(int), 0, h - 1)
            frame[yi, xi] = src_colors

            pygame.surfarray.blit_array(screen, frame.swapaxes(0, 1))


        # Top bar
        pygame.draw.rect(screen, (30, 30, 30), (0, 0, w, TOPBAR_HEIGHT))

        dropdown.draw(screen, font)
        start_btn.draw(screen, font)


        pygame.display.flip()
        clock.tick(FRAME_RATE)

    pygame.quit()

if __name__ == "__main__":
    main()
