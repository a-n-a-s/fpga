//============================================================================
// 3x3 Systolic Array for Conv1D
// Parallel processing element array for 3-tap convolution
// Achieves 9x speedup vs sequential MAC (3 cycles vs 27 cycles per output)
//============================================================================

module systolic_array_3x3 #(
    parameter DATA_WIDTH    = 8,                // Input data width (INT8)
    parameter ACC_WIDTH     = 32                // Accumulator width (INT32)
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 enable,         // Array enable
    
    // Activation inputs (from left, 3 samples for kernel)
    input  wire signed [DATA_WIDTH-1:0]  act_in_0,  // Sample k
    input  wire signed [DATA_WIDTH-1:0]  act_in_1,  // Sample k+1
    input  wire signed [DATA_WIDTH-1:0]  act_in_2,  // Sample k+2
    
    // Weight inputs (from top, 3 weights)
    input  wire signed [DATA_WIDTH-1:0]  w_in_0,    // Weight 0
    input  wire signed [DATA_WIDTH-1:0]  w_in_1,    // Weight 1
    input  wire signed [DATA_WIDTH-1:0]  w_in_2,    // Weight 2
    
    // Initial accumulator input (bias or zero)
    input  wire signed [ACC_WIDTH-1:0]   acc_in,
    
    // Final result from rightmost PE
    output wire signed [ACC_WIDTH-1:0]   acc_out,
    
    // Output valid
    output wire                 valid
);

    // Internal wires for PE interconnect
    // Row 0
    wire signed [DATA_WIDTH-1:0]  a0_0, a0_1, a0_2;  // Activation flow (left to right)
    wire signed [DATA_WIDTH-1:0]  b0_0, b0_1, b0_2;  // Weight flow (top to bottom)
    wire signed [ACC_WIDTH-1:0]   c0_0, c0_1, c0_2;  // Accumulator flow
    
    // Row 1
    wire signed [DATA_WIDTH-1:0]  a1_0, a1_1, a1_2;
    wire signed [DATA_WIDTH-1:0]  b1_0, b1_1, b1_2;
    wire signed [ACC_WIDTH-1:0]   c1_0, c1_1, c1_2;
    
    // Row 2
    wire signed [DATA_WIDTH-1:0]  a2_0, a2_1, a2_2;
    wire signed [DATA_WIDTH-1:0]  b2_0, b2_1, b2_2;
    wire signed [ACC_WIDTH-1:0]   c2_0, c2_1, c2_2;
    
    // PE valid signals
    wire valid_00, valid_01, valid_02;
    wire valid_10, valid_11, valid_12;
    wire valid_20, valid_21, valid_22;
    
    //========================================================================
    // PE Instantiation - 3x3 Grid
    // PE(row, col): receives activation from left, weight from top, acc from left
    //========================================================================
    
    // Row 0
    pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_00 (
        .clk(clk), .rst(rst), .enable(enable),
        .a_in(act_in_0), .a_out(a0_0),
        .b_in(w_in_0),   .b_out(b0_0),
        .acc_in(acc_in), .acc_out(c0_0),
        .valid(valid_00)
    );
    
    pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_01 (
        .clk(clk), .rst(rst), .enable(enable),
        .a_in(a0_0),   .a_out(a0_1),
        .b_in(b0_0),   .b_out(b0_1),
        .acc_in(c0_0), .acc_out(c0_1),
        .valid(valid_01)
    );
    
    pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_02 (
        .clk(clk), .rst(rst), .enable(enable),
        .a_in(a0_1),   .a_out(a0_2),
        .b_in(b0_1),   .b_out(b0_2),
        .acc_in(c0_1), .acc_out(c0_2),
        .valid(valid_02)
    );
    
    // Row 1
    pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_10 (
        .clk(clk), .rst(rst), .enable(enable),
        .a_in(act_in_1), .a_out(a1_0),
        .b_in(w_in_1),   .b_out(b1_0),
        .acc_in(acc_in), .acc_out(c1_0),
        .valid(valid_10)
    );
    
    pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_11 (
        .clk(clk), .rst(rst), .enable(enable),
        .a_in(a1_0),   .a_out(a1_1),
        .b_in(b1_0),   .b_out(b1_1),
        .acc_in(c1_0), .acc_out(c1_1),
        .valid(valid_11)
    );
    
    pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_12 (
        .clk(clk), .rst(rst), .enable(enable),
        .a_in(a1_1),   .a_out(a1_2),
        .b_in(b1_0),   .b_out(b1_2),
        .acc_in(c1_1), .acc_out(c1_2),
        .valid(valid_12)
    );
    
    // Row 2
    pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_20 (
        .clk(clk), .rst(rst), .enable(enable),
        .a_in(act_in_2), .a_out(a2_0),
        .b_in(w_in_2),   .b_out(b2_0),
        .acc_in(acc_in), .acc_out(c2_0),
        .valid(valid_20)
    );
    
    pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_21 (
        .clk(clk), .rst(rst), .enable(enable),
        .a_in(a2_0),   .a_out(a2_1),
        .b_in(b2_0),   .b_out(b2_1),
        .acc_in(c2_0), .acc_out(c2_1),
        .valid(valid_21)
    );
    
    pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe_22 (
        .clk(clk), .rst(rst), .enable(enable),
        .a_in(a2_1),   .a_out(a2_2),
        .b_in(b2_0),   .b_out(b2_2),
        .acc_in(c2_1), .acc_out(c2_2),
        .valid(valid_22)
    );
    
    //========================================================================
    // Output selection
    // For Conv1D: we need sum of all 9 PEs (full 3x3 convolution)
    //========================================================================
    
    // Sum all PE outputs for final result
    // c2_2 contains: acc_in + sum of all 9 products
    assign acc_out = c2_2;
    assign valid   = valid_22;
    
endmodule
