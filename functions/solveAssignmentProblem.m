function [assignment] = solveAssignmentProblem(costmatrix)
% solveAssignmentProblem: 
% the assignment problem involves assigning tasks to workers with minimum cost or maximum benefit

    [n, m] = size(costmatrix); % Number of workers (n) and tasks (m)

    f = reshape(costmatrix', n*m, 1); % Linearize the cost matrix

    intcon = 1:(n*m); % Integer constraints for decision variables

    Aeq = kron(eye(n), ones(1, m)); % Constraint: Each worker performs only one task
    beq = ones(n, 1);

    Aeq = [Aeq; kron(ones(1, n), eye(m))]; % Constraint: Each task is assigned to only one worker
    beq = [beq; ones(m, 1)];

    lb = zeros(n*m, 1); % Lower bounds: decision variables are binary (0 or 1)
    ub = ones(n*m, 1); % Upper bounds: decision variables are binary (0 or 1)

    options = optimoptions('intlinprog', 'Display', 'off'); % Set options

    [x, ~, ~] = intlinprog(f, intcon, [], [], Aeq, beq, lb, ub, options); % 'intlinprog' function is used to solve the Integer Linear Programming problem.

    % Reshape the solution vector into a matrix
    assignment = reshape(x, m, n)';
end

% costmatrix = [10, 7, 8, 5;
%               5, 6, 8, 9;
%               9, 12, 6, 4;
%               3, 5, 4, 7];
% 
% costmatrix = [426, 426, 426;
%               389, 389, 389;
%               391, 391, 391];
% 
% [n, m] = size(costmatrix); % Number of workers (n) and tasks (m)
% 
% f = reshape(costmatrix', n*m, 1); % Linearize the cost matrix
% 
% intcon = 1:(n*m); % Integer constraints for decision variables
% 
% Aeq = kron(eye(n), ones(1, m)); % Constraint: Each worker performs only one task
% beq = ones(n, 1);
% 
% Aeq = [Aeq; kron(ones(1, n), eye(m))]; % Constraint: Each task is assigned to only one worker
% beq = [beq; ones(m, 1)];
% 
% lb = zeros(n*m, 1); % Lower bounds: decision variables are binary (0 or 1)
% ub = ones(n*m, 1); % Upper bounds: decision variables are binary (0 or 1)
% 
% options = optimoptions('intlinprog', 'Display', 'off'); % Set options
% 
% [x, fval, exitflag] = intlinprog(f, intcon, [], [], Aeq, beq, lb, ub, options);
% 
% % Reshape the solution vector into a matrix
% assignment = reshape(x, m, n)';
% 
% % Display the assignment matrix
% disp('Assignment Matrix:');
% disp(assignment);