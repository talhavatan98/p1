import pygame 
import matplotlib.pyplot as plt
import numpy as np
from auv_tracking import UnderwaterSimulation

def plot_tracking_and_distance(real_positions, noisy_positions, kalman_positions, distances):

    time_steps = np.arange(len(real_positions))
    

    real_x, real_y = zip(*real_positions)
    noisy_x, noisy_y = zip(*noisy_positions)
    kalman_x, kalman_y = zip(*kalman_positions)
    
    plt.figure(figsize=(14, 6))
    

    plt.subplot(1, 2, 1)
    plt.plot(real_x, real_y, label='Gerçek Konum', color='blue', marker='o', linewidth=2)
    plt.plot(noisy_x, noisy_y, label='Gürültülü Ölçüm', color='orange', linestyle='--', alpha=0.7)
    plt.plot(kalman_x, kalman_y, label='Kalman Tahmini', color='green', marker='x')
    plt.title("Hedef ve İzleyici Konum Takibi", fontsize=14)
    plt.xlabel("X Konumu (px)")
    plt.ylabel("Y Konumu (px)")
    plt.legend()
    plt.grid(alpha=0.3)
    

    plt.subplot(1, 2, 2)
    plt.plot(time_steps, distances, label='Mesafe', color='red', linewidth=2)
    plt.title("Hedef ve İzleyici Arasındaki Mesafe", fontsize=14)
    plt.xlabel("Zaman Adımı")
    plt.ylabel("Mesafe (px)")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def run_simulation_with_analysis():

    sim = UnderwaterSimulation()
    

    real_positions = []
    noisy_positions = []
    kalman_positions = []
    distances = []
    
    sim.running = True
    while sim.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim.running = False

        if not sim.paused:
            sim.handle_input()


            sim.target.update(dt=1/60)
            sim.keep_in_bounds(sim.target)


            measurement = sim.add_noise(sim.target.pos)
            sim.kalman.predict()
            estimated_pos = sim.kalman.update(measurement)


            sim.tracker.update(estimated_pos.flatten(), dt=1/60)
            sim.keep_in_bounds(sim.tracker)


            real_positions.append(sim.target.pos.copy())
            noisy_positions.append(measurement)
            kalman_positions.append(estimated_pos.flatten())
            distances.append(np.linalg.norm(sim.target.pos - sim.tracker.pos))


        sim.screen.fill(sim.DARK_BLUE)
        sim.draw_particles()
        sim.target.draw(sim.screen)
        sim.tracker.draw(sim.screen)
        pygame.draw.circle(sim.screen, sim.YELLOW, estimated_pos.flatten().astype(int), 5)
        if sim.show_info:
            sim.draw_info()
        pygame.display.flip()
        sim.clock.tick(60)

    pygame.quit()


    plot_tracking_and_distance(real_positions, noisy_positions, kalman_positions, distances)

if __name__ == "__main__":
    run_simulation_with_analysis()