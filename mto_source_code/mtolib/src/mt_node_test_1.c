#include "mt_objects.h"

#include "gsl/gsl_cdf.h"
//#include <assert.h>

// Actual significance test
int mt_node_test_1(mt_object_data* mt_o, INT_TYPE index)
{
  // Get pointer to object data
  mt_data *mt = mt_o->mt;
  // Get significance level
  FLOAT_TYPE significance_level_power = mt_o->paras->alpha;
  
  // Set 'node' to point to index of node (and attr to attribute)
  mt_node* node = mt->nodes + index;
  mt_node_attributes* attr = mt->nodes_attributes + index;

  // Get ID of parent
  INT_TYPE parent_idx = node->parent;
  INT_TYPE area = node->area;

  FLOAT_TYPE min_val = mt->nodes_attributes[parent_idx].min_branch_val;
  FLOAT_TYPE min_val2 = min_val * min_val;
  FLOAT_TYPE new_variance = (min_val / mt_o->paras->gain) + mt_o->paras->bg_variance;
  FLOAT_TYPE new_power = (attr->sumsq + (area*min_val2) - (2*min_val*attr->sum));
  FLOAT_TYPE new_power_normalized = new_power / new_variance;
  
  // gsl_cdf_chisq_Q doesn't like areas that are too large.
  if (area > 256*256)
  {
     new_power_normalized /= area;
     area = 256*256;
     new_power_normalized *= area;
  }

  return gsl_cdf_chisq_Q(new_power_normalized, area) <
    significance_level_power;
}

void mt_node_test_1_data_free(mt_object_data* mt_o)
{
  free(mt_o->node_significance_test_data);

  mt_o->node_significance_test_data = NULL;
}

// Main
void mt_use_node_test_1(mt_object_data* mt_o)
  //FLOAT_TYPE significance_level_power)
{
  // Check verbosity
  if (mt_o->mt->verbosity_level)
  {
    printf("Using significance test 1 (power given area).\n");
  }

  // Set significance test to this
  mt_o->node_significance_test = mt_node_test_1;

  // If there is already test data, clear it
  if (mt_o->node_significance_test_data != NULL)
  {
    mt_o->node_significance_test_data_free(mt_o);
  }

  // Allocate memory for test data
  mt_o->node_significance_test_data = malloc(sizeof(FLOAT_TYPE));

  // Set significance level?
  //*((FLOAT_TYPE *)mt_o->node_significance_test_data) =
  //  significance_level_power;

  // Free significance test data memory
  mt_o->node_significance_test_data_free = mt_node_test_1_data_free;
}
