//============================================================================
// Processing Element (PE) for Systolic Array
// Performs: acc = acc + (a * b) in single cycle
// Data flows: left-to-right (a), top-to-bottom (b), accumulation stays local
//============================================================================

module pe #(
    parameter DATA_WIDTH    = 8,                // Input data width (INT8)
    parameter ACC_WIDTH     = 32                // Accumulator width (INT32)
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 enable,         // PE enable signal
    
    // Data input from left (activation)
    input  wire signed [DATA_WIDTH-1:0]  a_in,
    output wire signed [DATA_WIDTH-1:0]  a_out,
    
    // Weight input from top
    input  wire signed [DATA_WIDTH-1:0]  b_in,
    output wire signed [DATA_WIDTH-1:0]  b_out,
    
    // Accumulator input from left neighbor
    input  wire signed [ACC_WIDTH-1:0]   acc_in,
    output wire signed [ACC_WIDTH-1:0]   acc_out,
    
    // Output valid
    output reg                  valid
);

    // Internal registers
    reg signed [ACC_WIDTH-1:0]  accumulator;
    reg signed [ACC_WIDTH-1:0]  product;
    reg                         valid_reg;
    
    // Sign-extend inputs for multiplication
    wire signed [ACC_WIDTH-1:0] a_extended;
    wire signed [ACC_WIDTH-1:0] b_extended;
    
    assign a_extended = {{(ACC_WIDTH-DATA_WIDTH){a_in[DATA_WIDTH-1]}}, a_in};
    assign b_extended = {{(ACC_WIDTH-DATA_WIDTH){b_in[DATA_WIDTH-1]}}, b_in};
    
    //========================================================================
    // Single-cycle MAC with registered accumulation
    //========================================================================
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            accumulator <= {ACC_WIDTH{1'b0}};
            product     <= {ACC_WIDTH{1'b0}};
            valid_reg   <= 1'b0;
        end
        else if (enable) begin
            // Multiply-accumulate
            product     <= a_extended * b_extended;
            accumulator <= acc_in + product;
            valid_reg   <= 1'b1;
        end
        else begin
            // Hold: pass through accumulator input
            accumulator <= acc_in;
            valid_reg   <= 1'b0;
        end
    end
    
    // Combinational outputs (pass through)
    assign a_out    = a_in;
    assign b_out    = b_in;
    assign acc_out  = accumulator;
    assign valid    = valid_reg;
    
endmodule
