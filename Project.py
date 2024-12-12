import numpy as np
import matplotlib.pyplot as plt
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import roboticstoolbox as rtb
from spatialmath import SE3

def calculate_valid_height_range(L1, L2, head_to_waist):
    """
    Calculate the valid height range for the robot based on link lengths.
    
    Args:
    L1 (float): Length of first link
    L2 (float): Length of second link
    head_to_waist (float): Distance from waist to head
    
    Returns:
    tuple: (min_height, max_height)
    """
    # Maximum possible height (fully extended)
    max_height = L1 + L2 + head_to_waist
    
    # Minimum possible height (finding the lowest point where joint angles are valid)
    min_height = head_to_waist  # Start with head_to_waist as initial minimum
    
    for height in range(int(head_to_waist), int(max_height) + 1):
        try:
            # Calculate joint angles to check validity
            beta_test = np.arccos((L1**2 + L2**2 - (height - head_to_waist)**2) / (2 * L1 * L2))
            q2_test = np.pi - beta_test
            
            x_h_test = L2 + L1 * np.cos(q2_test)
            y_h_test = L1 * np.sin(q2_test)
            
            out_theta_test = np.arctan2(x_h_test, y_h_test)
            alpha_test = np.pi/2 - out_theta_test
            
            # Check if angle is within a reasonable range
            if np.rad2deg(alpha_test) < 90:
                min_height = height
                break
        except Exception:
            # If calculation fails, continue to next iteration
            continue
    
    return min_height, max_height

def create_robot_legs(L1, L2):
    # Leg 1
    Leg1 = rtb.DHRobot(
        [
            rtb.RevoluteMDH(alpha=np.pi/2, offset=np.pi/2),
            rtb.RevoluteMDH(a=L2),
            rtb.RevoluteMDH(a=L1, offset=np.pi/2),
            rtb.RevoluteMDH(a=6, alpha=np.pi/2, offset=np.pi),
            rtb.RevoluteMDH(),
            rtb.RevoluteMDH(d=10),
            rtb.RevoluteMDH(alpha=np.pi/2),
            rtb.RevoluteMDH(a=5, alpha=-np.pi/2),
        ], base=SE3(6, 0, 0),
        name="Leg1"
    )

    # Leg 2
    Leg2 = rtb.DHRobot(
        [
            rtb.RevoluteMDH(alpha=np.pi/2, offset=np.pi/2),
            rtb.RevoluteMDH(a=L2),
            rtb.RevoluteMDH(a=L1),
            rtb.RevoluteMDH(alpha=-np.pi/3, offset=-np.pi/2),
            rtb.RevoluteMDH(a=6, alpha=-np.pi/2, offset=np.pi/3),
            rtb.RevoluteMDH(d=10),
            rtb.RevoluteMDH(alpha=np.pi/2),
            rtb.RevoluteMDH(a=5, alpha=-np.pi/2),
        ], base=SE3(-np.sin(np.deg2rad(30))*6, np.cos(np.deg2rad(30))*6, 0),
        name="Leg2"
    )

    # Leg 3
    Leg3 = rtb.DHRobot(
        [
            rtb.RevoluteMDH(alpha=np.pi/2, offset=np.pi/2),
            rtb.RevoluteMDH(a=L2),
            rtb.RevoluteMDH(a=L1),
            rtb.RevoluteMDH(alpha=np.pi/3, offset=-np.pi/2),
            rtb.RevoluteMDH(a=6, alpha=-np.pi/2, offset=-np.pi/3),
            rtb.RevoluteMDH(d=10),
            rtb.RevoluteMDH(alpha=np.pi/2),
            rtb.RevoluteMDH(a=5, alpha=-np.pi/2),
        ], base=SE3(-np.sin(np.deg2rad(30))*6, -np.cos(np.deg2rad(30))*6, 0),
        name="Leg3"
    )

    return Leg1, Leg2, Leg3

def calculate_angles(Height, L1, L2, Degree_of_head_tiit):
    h = Height - 10  # Head_to_Waist is 10
    # Avoid domain errors in arccos
    if h > (L1 + L2) or h < abs(L1 - L2):
        raise ValueError("Height is out of reachable range for the robot legs.")
    
    beta = np.arccos((L1**2 + L2**2 - h**2) / (2 * L1 * L2))
    q2 = np.pi - beta

    x_h = L2 + L1 * np.cos(q2)
    y_h = L1 * np.sin(q2)
    out_theta = np.arctan2(x_h, y_h)
    alpha = np.pi / 2 - out_theta
    q1 = -(np.pi / 2 - out_theta)

    gamma = np.pi - alpha - beta
    q3 = -gamma
    q7 = np.deg2rad(Degree_of_head_tiit)
    
    return q1, q2, q3, q7

def plot_robot(height, degree_of_head_tilt, L1, L2):
    Leg1, Leg2, Leg3 = create_robot_legs(L1, L2)
    q1, q2, q3, q7 = calculate_angles(height, L1, L2, degree_of_head_tilt)
    
    q_Leg1 = [q1, q2, q3, 0, 0, 0, q7, 0]
    q_Leg2 = [-q1, -q2, -q3, 0, 0, 0, q7, 0]
    q_Leg3 = [-q1, -q2, -q3, 0, 0, 0, q7, 0]

    def get_joint_positions(robot, q):
        T = robot.fkine_all(q)
        return np.array([T[i].t for i in range(len(T))])
    
    # Get joint positions for each leg
    positions_leg1 = get_joint_positions(Leg1, q_Leg1)
    positions_leg2 = get_joint_positions(Leg2, q_Leg2)
    positions_leg3 = get_joint_positions(Leg3, q_Leg3)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(8, 6))
    
    # 3D Plot (first subplot)
    ax = fig.add_subplot(221, projection="3d")
    ax.set_title("Leg1, Leg2, Leg3 Plots in One Figure")

    # Plot lines for each leg in 3D
    ax.plot(positions_leg1[:, 0], positions_leg1[:, 1], positions_leg1[:, 2], 'o-', label="Leg1", color='r')
    ax.plot(positions_leg2[:, 0], positions_leg2[:, 1], positions_leg2[:, 2], 'o-', label="Leg2", color='g')
    ax.plot(positions_leg3[:, 0], positions_leg3[:, 1], positions_leg3[:, 2], 'o-', label="Leg3", color='b')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.grid(True)

    # XZ Plane (second subplot)
    ax2 = fig.add_subplot(222)
    ax2.plot(positions_leg1[:, 0], positions_leg1[:, 2], 'o-', label="Leg1", color='r')
    ax2.plot(positions_leg2[:, 0], positions_leg2[:, 2], 'o-', label="Leg2", color='g')
    ax2.plot(positions_leg3[:, 0], positions_leg3[:, 2], 'o-', label="Leg3", color='b')
    ax2.set_title("XZ Plane")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.legend()
    ax2.grid(True)
    ax2.axis("equal")
        
    # YZ Plane (third subplot)
    ax3 = fig.add_subplot(223)
    ax3.plot(positions_leg1[:, 1], positions_leg1[:, 2], 'o-', label="Leg1", color='r')
    ax3.plot(positions_leg2[:, 1], positions_leg2[:, 2], 'o-', label="Leg2", color='g')
    ax3.plot(positions_leg3[:, 1], positions_leg3[:, 2], 'o-', label="Leg3", color='b')
    ax3.set_title("YZ Plane")
    ax3.set_xlabel("Y (m)")
    ax3.set_ylabel("Z (m)")
    ax3.legend()
    ax3.grid(True)
    ax3.axis("equal")

    # Adjust layout
    plt.tight_layout()

    # Convert to Pygame surface
    canvas = FigureCanvas(fig)
    canvas.draw()
    raw_data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    width, height = canvas.get_width_height()
    surface = pygame.image.frombuffer(raw_data, (width, height), "RGB")
    plt.close(fig)
    
    return surface, Leg1, Leg2, Leg3, q_Leg1, q_Leg2, q_Leg3

# Global parameters with initial values
HEAD_TO_WAIST = 10  # m
L1 = 15  # Length of first link
L2 = 10  # Length of second link
HEIGHT = 22  # Initial height
DEGREE_OF_HEAD_TILT = 30  # Initial head tilt

def main(L1, L2, head_to_waist):
    # Use these local variables instead of global
    height = HEIGHT
    degree_of_head_tilt = DEGREE_OF_HEAD_TILT
    
    # Dynamically calculate height range
    min_height, max_height = calculate_valid_height_range(L1, L2, head_to_waist)

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((1400, 850))  # Increased height to accommodate larger plot
    pygame.display.set_caption("Robot Real-Time Plot with Adjustable Parameters")
    font = pygame.font.SysFont(None, 36)
    input_font = pygame.font.SysFont(None, 28)

    # Initial parameters
    input_active = False
    input_text = ""
    param_selection = "Height"

    # Main game loop
    running = True
    while running:
        # Recalculate height range in case L1 or L2 changed
        min_height, max_height = calculate_valid_height_range(L1, L2, head_to_waist)
        
        # Ensure current height is within new range
        height = max(min_height, min(height, max_height))

        screen.fill((255, 255, 255))

        # Plot robot and get surface
        graph_surface, Leg1, Leg2, Leg3, q_Leg1, q_Leg2, q_Leg3 = plot_robot(height, degree_of_head_tilt, L1, L2)
        screen.blit(graph_surface, (50, 50))

        # Draw parameter input rectangle
        pygame.draw.rect(screen, (230, 230, 230), (900, 80, 400, 380), border_radius=10)
        pygame.draw.rect(screen, (200, 200, 200), (910, 100, 380, 40), border_radius=5)
        input_surface = input_font.render(input_text, True, (0, 0, 0))
        screen.blit(input_surface, (910, 110))

        # Render parameter texts
        text_height = font.render(f"Height: {height:.2f} meter", True, (0, 0, 0))
        text_tilt = font.render(f"Head Tilt: {degree_of_head_tilt} degree", True, (0, 0, 0))
        text_L1 = font.render(f"Link 1: {L1:.2f} meter", True, (0, 0, 0))
        text_L2 = font.render(f"Link 2: {L2:.2f} meter", True, (0, 0, 0))
        param_text = font.render(f"Editing {param_selection} :", True, (0, 0, 0))

        # Blit parameter texts
        screen.blit(text_height, (910, 180))
        screen.blit(text_tilt, (910, 220))
        screen.blit(text_L1, (910, 260))
        screen.blit(text_L2, (910, 300))
        screen.blit(param_text, (910, 340))

        # Instructions
        instruction = input_font.render("Use Tab to switch parameters", True, (0, 0, 0))
        screen.blit(instruction, (50, 680))
        instruction2 = input_font.render("Enter value and press Enter to update", True, (0, 0, 0))
        screen.blit(instruction2, (50, 710))
        instruction3 = input_font.render(f"Maximum Height: {max_height:.2f} meter", True, (0, 0, 0))
        screen.blit(instruction3, (910, 390))
        instruction4 = input_font.render(f"Minimum Hight: {min_height:.2f} meter", True, (0, 0, 0))
        screen.blit(instruction4, (910, 410))

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if input_active:
                    if event.key == pygame.K_RETURN:
                        try:
                            new_value = float(input_text)
                        
                            if param_selection == "Height":
                                height = max(min_height, min(new_value, max_height))
                            elif param_selection == "Degree_of_head_tiit":
                                degree_of_head_tilt = new_value
                            elif param_selection == "L1":
                                L1 = new_value
                            elif param_selection == "L2":
                                L2 = new_value
                            input_text = ""
                        except ValueError:
                            input_text = ""
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode
                elif event.key == pygame.K_TAB:
                    if param_selection == "Height":
                        param_selection = "Degree_of_head_tiit"
                    elif param_selection == "Degree_of_head_tiit":
                        param_selection = "L1"
                    elif param_selection == "L1":
                        param_selection = "L2"
                    else:
                        param_selection = "Height"
            if event.type == pygame.MOUSEBUTTONDOWN:
                if 910 <= event.pos[0] <= 1140 and 100 <= event.pos[1] <= 140:
                    input_active = True
                else:
                    input_active = False
        
        pygame.display.flip()

    pygame.quit()

# Call main with initial parameter values
if __name__ == "__main__":
    main(L1, L2, HEAD_TO_WAIST)