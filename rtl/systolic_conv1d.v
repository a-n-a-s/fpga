//============================================================================
// Linear Systolic Array for Conv1D (3 PEs)
// Optimized for 1D convolution with kernel size 3
// Each output computed in 3 cycles (vs 3 sequential MACs)
//============================================================================

module systolic_conv1d #(
    parameter DATA_WIDTH    = 8,                // Input data width (INT8)
    parameter ACC_WIDTH     = 32,               // Accumulator width (INT32)
    parameter KERNEL_SIZE   = 3                 // Convolution kernel size
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,          // Start convolution
    output reg                  done,           // Convolution complete
    
    // Input samples (3 consecutive samples for kernel)
    input  wire signed [DATA_WIDTH-1:0] sample_0,  // Sample at position k
    input  wire signed [DATA_WIDTH-1:0] sample_1,  // Sample at position k+1
    input  wire signed [DATA_WIDTH-1:0] sample_2,  // Sample at position k+2
    
    // Weights for current filter
    input  wire signed [DATA_WIDTH-1:0] weight_0,  // Weight 0
    input  wire signed [DATA_WIDTH-1:0] weight_1,  // Weight 1
    input  wire signed [DATA_WIDTH-1:0] weight_2,  // Weight 2
    
    // Bias for current filter
    input  wire signed [ACC_WIDTH-1:0]  bias,
    
    // Result
    output reg  signed [ACC_WIDTH-1:0]  result,
    output reg                          valid
);

    // Internal wires for PE interconnect
    wire signed [DATA_WIDTH-1:0]  a0, a1;       // Activation flow
    wire signed [DATA_WIDTH-1:0]  b0, b1;       // Weight flow  
    wire signed [ACC_WIDTH-1:0]   c0, c1, c2;   // Accumulator flow
    
    // PE outputs
    wire signed [ACC_WIDTH-1:0]   pe0_acc, pe1_acc, pe2_acc;
    wire                          pe0_valid, pe1_valid, pe2_valid;
    
    // State machine
    reg [1:0] state;
    localparam S_IDLE   = 2'd0;
    localparam S_LOAD   = 2'd1;
    localparam S_COMPUTE = 2'd2;
    localparam S_OUTPUT = 2'd3;
    
    reg enable;
    
    //========================================================================
    // PE Instantiation - Linear Array (3 PEs)
    //========================================================================
    
    // PE 0: First tap
    pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe0 (
        .clk(clk), .rst(rst), .enable(enable),
        .a_in(sample_0), .a_out(a0),
        .b_in(weight_0), .b_out(b0),
        .acc_in(bias),   .acc_out(pe0_acc),
        .valid(pe0_valid)
    );
    
    // PE 1: Second tap
    pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe1 (
        .clk(clk), .rst(rst), .enable(enable),
        .a_in(a0),       .a_out(a1),
        .b_in(b0),       .b_out(b1),
        .acc_in(pe0_acc), .acc_out(pe1_acc),
        .valid(pe1_valid)
    );
    
    // PE 2: Third tap (final)
    pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) pe2 (
        .clk(clk), .rst(rst), .enable(enable),
        .a_in(a1),        .a_out(),
        .b_in(b1),        .b_out(),
        .acc_in(pe1_acc), .acc_out(pe2_acc),
        .valid(pe2_valid)
    );
    
    //========================================================================
    // Control State Machine
    //========================================================================
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state   <= S_IDLE;
            done    <= 1'b0;
            enable  <= 1'b0;
            result  <= {ACC_WIDTH{1'b0}};
            valid   <= 1'b0;
        end
        else begin
            done    <= 1'b0;
            valid   <= 1'b0;
            
            case (state)
                S_IDLE: begin
                    if (start) begin
                        enable  <= 1'b1;
                        state   <= S_COMPUTE;
                    end
                end
                
                S_COMPUTE: begin
                    // After one cycle, result is ready at PE2
                    if (pe2_valid) begin
                        result  <= pe2_acc;
                        valid   <= 1'b1;
                        done    <= 1'b1;
                        enable  <= 1'b0;
                        state   <= S_IDLE;
                    end
                end
                
                default: state <= S_IDLE;
            endcase
        end
    end
    
endmodule
