import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from auv_tracking import UnderwaterSimulation
import pygame

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    transform = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(np.mean(x), np.mean(y))
    
    ellipse.set_transform(transform + ax.transData)
    return ax.add_patch(ellipse)

def collect_simulation_data(duration_seconds=10):
    sim = UnderwaterSimulation()
    
    target_positions = []
    tracker_positions = []
    estimated_positions = []
    particles = []
    
    start_time = pygame.time.get_ticks()
    running = True
    

    sim.target.vel = np.random.randn(2) * 2
    
    while running and (pygame.time.get_ticks() - start_time) < duration_seconds * 1000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        

        sim.target.vel += np.random.randn(2) * 0.1
        sim.target.vel = np.clip(sim.target.vel, -sim.target.speed, sim.target.speed)
        sim.target.pos += sim.target.vel
        

        sim.target.pos[0] = np.clip(sim.target.pos[0], 0, sim.width)
        sim.target.pos[1] = np.clip(sim.target.pos[1], 0, sim.height)
        

        measurement = sim.add_noise(sim.target.pos)
        sim.kalman.predict()
        estimated_pos = sim.kalman.update(measurement)
        

        target_positions.append(sim.target.pos.copy())
        tracker_positions.append(sim.tracker.pos.copy())
        estimated_positions.append(estimated_pos.flatten())
        particles.append([(p[0], p[1]) for p in sim.particles])
        

        sim.tracker.update(estimated_pos.flatten(), dt=1/60)
        

        sim.screen.fill(sim.DARK_BLUE)
        sim.target.draw(sim.screen)
        sim.tracker.draw(sim.screen)
        pygame.display.flip()
        sim.clock.tick(60)
    
    pygame.quit()
    
    if len(target_positions) > 0:
        return (np.array(target_positions), 
                np.array(tracker_positions),
                np.array(estimated_positions), 
                particles)
    else:
        return None

def create_analysis_plots():

    data = collect_simulation_data()
    if data is None:
        print("Veri toplanamadı!")
        return
    
    target_pos, tracker_pos, estimated_pos, particles = data
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    

    ax1.set_title('Konum Takip ve Kovaryans Elipsleri')
    ax1.plot(target_pos[:, 0], target_pos[:, 1], 'r.-', label='Hedef', alpha=0.5)
    ax1.plot(tracker_pos[:, 0], tracker_pos[:, 1], 'g.-', label='Takipçi', alpha=0.5)
    ax1.plot(estimated_pos[:, 0], estimated_pos[:, 1], 'b.-', label='Tahmin', alpha=0.5)
    

    step = 50
    for i in range(0, len(target_pos), step):
        confidence_ellipse(estimated_pos[i:i+step, 0], estimated_pos[i:i+step, 1],
                         ax1, n_std=2, alpha=0.1, color='blue')
    
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlabel('X Konumu')
    ax1.set_ylabel('Y Konumu')
    

    ax2.set_title('Parçacık Dağılımı')
    particle_positions = np.array(particles[0])
    x_coords = [p[0] for p in particle_positions]
    y_coords = [p[1] for p in particle_positions]
    ax2.scatter(x_coords, y_coords, c='blue', alpha=0.5, s=10)
    ax2.plot(target_pos[0, 0], target_pos[0, 1], 'r*', 
             label='Başlangıç Konumu', markersize=15)
    
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlabel('X Konumu')
    ax2.set_ylabel('Y Konumu')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_analysis_plots()