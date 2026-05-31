clc; 
clear; 
close all;

% Define Parameters
VDD = 1.1; % Supply Voltage
Vin = linspace(0, VDD, 1000); % Sweep Input Voltage from 0V to VDD
Vth_NMOS = 0.4; % NMOS threshold voltage
Vth_PMOS = -0.4; % PMOS threshold voltage

% Initialize Output Voltage
Vout = zeros(size(Vin));

% Compute Output Voltage for DC Sweep
for i = 1:length(Vin)
    if Vin(i) < abs(Vth_PMOS) % PMOS ON, NMOS OFF
        Vout(i) = 0; % PMOS pulls output LOW
    elseif Vin(i) > Vth_NMOS % NMOS ON, PMOS OFF
        Vout(i) = VDD; % NMOS pulls output HIGH
    else
        % Transition region where both transistors conduct
        Vout(i) = VDD * ((Vin(i) - Vth_NMOS) / (VDD - Vth_NMOS));  
    end
end

% Plot the DC Sweep Response
figure;
plot(Vin, Vout, 'b', 'LineWidth', 2); 
hold on;

% Add Midpoint Voltage Line
Vm = VDD / 2; % Midpoint Voltage
yline(Vm, 'r--', 'LineWidth', 1.5);
xline(Vm, 'r--', 'LineWidth', 1.5);

% Formatting
xlabel('Input Voltage (V)');
ylabel('Output Voltage (V)');
title('DC Sweep Analysis of NMOS-Top, PMOS-Bottom Circuit');
legend('DC Sweep Response', 'Midpoint Voltage', 'Location', 'northeast');
grid on;
xlim([0, VDD]);
ylim([0, VDD]);

% Display the plot
hold off;
