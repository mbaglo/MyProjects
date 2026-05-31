% Define the range of input voltages (Vin) from 0V to VDD (1.1V)
VDD = 1.1; % Supply voltage
Vin = linspace(0, VDD, 1000); 

% Define the switching threshold (midpoint voltage)
Vm = VDD / 2; 

% Approximate inverter Vout behavior using a steep transition function
Vout = VDD ./ (1 + exp(30 * (Vin - Vm))); 

% Plot the VTC curve
figure;
plot(Vin, Vout, 'b', 'LineWidth', 2); % Plot Vout vs. Vin
hold on;

% Highlight the midpoint voltage with a vertical dashed red line
plot([Vm, Vm], [0, VDD], 'r--', 'LineWidth', 1.5); 

% Highlight the ideal switching level with a horizontal dashed red line
plot([0, VDD], [Vm, Vm], 'r--', 'LineWidth', 1.5); 

% Labels and formatting
xlabel('Input Voltage (V)'); % Label X-axis
ylabel('Output Voltage (V)'); % Label Y-axis
title('CMOS Inverter Voltage Transfer Characteristic'); % Plot title
grid on; % Enable grid
legend('VTC Curve', 'Midpoint Voltage'); % Add legend
xlim([0 VDD]); % Set X-axis limits
ylim([0 VDD]); % Set Y-axis limits

% Display the plot
hold off;
