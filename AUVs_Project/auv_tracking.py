import pygame
import numpy as np
from numpy.linalg import inv
import math

class KalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.B = np.array([[0], [0], [0], [0]])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 5
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 1000

    def predict(self, u=None):
        if u is None:
            u = np.zeros((1, 1))
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:2]

    def update(self, measurement):
        measurement = measurement.reshape((2, 1))
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2]

class Vehicle:
    def __init__(self, x, y, color, size=15, speed=5, vehicle_type="default"):
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([0.0, 0.0])
        self.color = color
        self.size = size
        self.speed = speed
        self.trail = []
        self.max_trail = 30
        self.angle = 0
        self.vehicle_type = vehicle_type
        self.energy = 100
        self.health = 100
        self.boost_cooldown = 0
        self.shield_active = False
        self.shield_cooldown = 0

    def update(self, target_pos=None, dt=1/60):
        # Update boost cooldown
        if self.boost_cooldown > 0:
            self.boost_cooldown -= dt
        if self.shield_cooldown > 0:
            self.shield_cooldown -= dt
            if self.shield_cooldown <= 0:
                self.shield_active = False

        # Energy regeneration
        self.energy = min(100, self.energy + 5 * dt)

        if target_pos is not None:
            direction = target_pos - self.pos
            distance = np.linalg.norm(direction)
            if distance > 0:
                self.vel = (direction / distance) * self.speed
                self.angle = math.atan2(direction[1], direction[0])

        self.pos += self.vel
        self.trail.append(self.pos.copy())
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

    def boost(self):
        if self.energy >= 30 and self.boost_cooldown <= 0:
            self.speed *= 2
            self.energy -= 30
            self.boost_cooldown = 3
            return True
        return False

    def activate_shield(self):
        if self.energy >= 40 and self.shield_cooldown <= 0:
            self.shield_active = True
            self.energy -= 40
            self.shield_cooldown = 5
            return True
        return False

    def draw(self, screen):
        # Draw trail
        if len(self.trail) > 1:
            for i in range(len(self.trail) - 1):
                alpha = int(255 * (i / len(self.trail)))
                trail_color = (*self.color[:3], alpha)
                pygame.draw.line(screen, trail_color,
                               self.trail[i], self.trail[i + 1], 2)

        if self.shield_active:
            pygame.draw.circle(screen, (100, 200, 255, 100),
                             self.pos.astype(int), self.size + 10, 2)


        if self.vehicle_type == "target":
            points = self.get_triangle_points()
            pygame.draw.polygon(screen, self.color, points)
        elif self.vehicle_type == "tracker":
            pygame.draw.circle(screen, self.color, self.pos.astype(int), self.size)
            end_pos = self.pos + np.array([math.cos(self.angle), math.sin(self.angle)]) * self.size
            pygame.draw.line(screen, self.color, self.pos, end_pos, 3)

    def get_triangle_points(self):
        angle = math.atan2(self.vel[1], self.vel[0])
        p1 = self.pos + np.array([math.cos(angle), math.sin(angle)]) * self.size
        p2 = self.pos + np.array([math.cos(angle + 2.6), math.sin(angle + 2.6)]) * self.size
        p3 = self.pos + np.array([math.cos(angle - 2.6), math.sin(angle - 2.6)]) * self.size
        return [p1, p2, p3]

class UnderwaterSimulation:
    def __init__(self):
        pygame.init()
        self.width = 1024
        self.height = 768
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Advanced Underwater Vehicle Tracking")
        

        self.BLUE = (0, 0, 255, 255)
        self.GREEN = (0, 255, 0, 255)
        self.RED = (255, 0, 0, 255)
        self.YELLOW = (255, 255, 0, 255)
        self.WHITE = (255, 255, 255, 255)
        self.DARK_BLUE = (0, 0, 50)
        
        self.clock = pygame.time.Clock()
        self.running = True
        

        self.target = Vehicle(self.width//2, self.height//2, self.RED, 20, 4, "target")
        self.tracker = Vehicle(self.width//4, self.height//4, self.GREEN, 15, 3, "tracker")
        

        self.kalman = KalmanFilter(dt=1/60)
        self.kalman.x = np.array([[self.target.pos[0]], 
                                 [self.target.pos[1]], 
                                 [0], 
                                 [0]])


        self.manual_control = False
        self.show_info = True
        self.paused = False
        

        self.particles = [(np.random.randint(0, self.width),
                          np.random.randint(0, self.height))
                         for _ in range(100)]
        

        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def add_noise(self, pos):
        return pos + np.random.normal(0, 2, 2)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        
        if self.manual_control:
            self.target.vel = np.array([0.0, 0.0])
            if keys[pygame.K_LEFT]:
                self.target.vel[0] = -self.target.speed
            if keys[pygame.K_RIGHT]:
                self.target.vel[0] = self.target.speed
            if keys[pygame.K_UP]:
                self.target.vel[1] = -self.target.speed
            if keys[pygame.K_DOWN]:
                self.target.vel[1] = self.target.speed
            

            if keys[pygame.K_SPACE]:
                if self.target.boost():
                    print("Boost activated!")
            

            if keys[pygame.K_s]:
                if self.target.activate_shield():
                    print("Shield activated!")
        else:

            self.target.vel += np.random.normal(0, 0.1, 2)
            self.target.vel = np.clip(self.target.vel, -self.target.speed, self.target.speed)

    def keep_in_bounds(self, vehicle):

        damping = 0.8
        if vehicle.pos[0] < 0:
            vehicle.pos[0] = 0
            vehicle.vel[0] *= -damping
        elif vehicle.pos[0] > self.width:
            vehicle.pos[0] = self.width
            vehicle.vel[0] *= -damping
            
        if vehicle.pos[1] < 0:
            vehicle.pos[1] = 0
            vehicle.vel[1] *= -damping
        elif vehicle.pos[1] > self.height:
            vehicle.pos[1] = self.height
            vehicle.vel[1] *= -damping

    def draw_info(self):
        texts = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Manual Control: {self.manual_control}",
            f"Target Energy: {int(self.target.energy)}%",
            f"Target Health: {int(self.target.health)}%",
            "Controls:",
            "M - Toggle manual/auto",
            "Arrow keys - Control target",
            "SPACE - Boost",
            "S - Shield",
            "P - Pause",
            "ESC - Quit",
            "I - Toggle info"
        ]
        
        for i, text in enumerate(texts):
            surface = self.font.render(text, True, self.WHITE)
            self.screen.blit(surface, (10, 10 + i * 30))

    def draw_particles(self):
        for i, (x, y) in enumerate(self.particles):

            y = (y - 0.5) % self.height
            self.particles[i] = (x, y)

            alpha = 128 + np.random.randint(-30, 30)
            color = (200, 200, 255, alpha)
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 1)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_m:
                        self.manual_control = not self.manual_control
                    elif event.key == pygame.K_i:
                        self.show_info = not self.show_info
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused

            if not self.paused:
                self.handle_input()
                

                self.target.update(dt=1/60)
                self.keep_in_bounds(self.target)
                

                measurement = self.add_noise(self.target.pos)
                self.kalman.predict()
                estimated_pos = self.kalman.update(measurement)
                

                self.tracker.update(estimated_pos.flatten(), dt=1/60)
                self.keep_in_bounds(self.tracker)


            self.screen.fill(self.DARK_BLUE)
            

            self.draw_particles()
            

            self.target.draw(self.screen)
            self.tracker.draw(self.screen)
            

            pygame.draw.circle(self.screen, self.YELLOW, 
                             estimated_pos.flatten().astype(int), 5)

            if self.show_info:
                self.draw_info()

            if self.paused:
                pause_text = self.font.render("PAUSED", True, self.WHITE)
                text_rect = pause_text.get_rect(center=(self.width/2, self.height/2))
                self.screen.blit(pause_text, text_rect)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    sim = UnderwaterSimulation()
    sim.run()
