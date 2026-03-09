//============================================================================
// MAC Unit - Sequential Multiply-Accumulate
// Performs: acc = acc + (a * b) over multiple cycles
// Synthesizable for Vivado FPGA implementation
//============================================================================

module mac_unit #(
    parameter DATA_WIDTH    = 8,      // Input data width (INT8)
    parameter ACC_WIDTH     = 32,     // Accumulator width (INT32)
    parameter NUM_STEPS     = 3       // Number of multiply-accumulate steps
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,          // Start MAC operation
    output reg                  done,           // MAC operation complete
    
    input  wire signed [DATA_WIDTH-1:0] data_in,    // Streaming data input
    input  wire signed [DATA_WIDTH-1:0] weight_in,  // Streaming weight input
    input  wire                 data_valid,         // Input data valid
    
    output reg  signed [ACC_WIDTH-1:0]  acc_out     // Final accumulator output
);

    // Internal registers
    reg signed [ACC_WIDTH-1:0]  accumulator;
    reg signed [ACC_WIDTH-1:0]  product;
    reg [7:0]                   step_cnt;
    reg                         running;
    reg                         data_valid_d;

    // State encoding
    localparam IDLE   = 1'b0;
    localparam RUN    = 1'b1;

    //========================================================================
    // Sequential MAC operation
    //========================================================================
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            accumulator   <= {ACC_WIDTH{1'b0}};
            product       <= {ACC_WIDTH{1'b0}};
            step_cnt      <= 8'd0;
            running       <= 1'b0;
            done          <= 1'b0;
            acc_out       <= {ACC_WIDTH{1'b0}};
            data_valid_d  <= 1'b0;
        end
        else begin
            done          <= 1'b0;  // Default: clear done
            data_valid_d  <= data_valid;
            
            if (start && !running) begin
                // Start new MAC operation
                running     <= 1'b1;
                step_cnt    <= 8'd0;
                accumulator <= {ACC_WIDTH{1'b0}};
            end
            else if (running) begin
                if (data_valid && step_cnt < NUM_STEPS) begin
                    // Perform multiply-accumulate
                    product     <= {{(ACC_WIDTH-DATA_WIDTH){data_in[DATA_WIDTH-1]}}, data_in} * 
                                   {{(ACC_WIDTH-DATA_WIDTH){weight_in[DATA_WIDTH-1]}}, weight_in};
                    accumulator <= accumulator + product;
                    step_cnt    <= step_cnt + 8'd1;
                end
                
                // Check if all steps completed
                if (step_cnt == NUM_STEPS - 1) begin
                    running   <= 1'b0;
                    done      <= 1'b1;
                    acc_out   <= accumulator + product;
                end
            end
            else begin
                // Hold output
                acc_out <= acc_out;
            end
        end
    end

endmodule
